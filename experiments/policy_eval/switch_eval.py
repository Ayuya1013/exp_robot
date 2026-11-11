"""待機中のどのフレームで、モデルの予測が「とどまる」から「動く」に切り替わるかを調べ、
その前後の画像(top, wrist)を書き出す。LeRobot 0.6.2(eb7a54c6)の policy_server と同じ手順で推論する。

使い方(研究室Linuxサーバー):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python switch_eval.py
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python switch_eval.py --episodes 0 6 12 17 --step 5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

BASE = Path.home() / "exp/robot/lerobot/outputs/train"
p = argparse.ArgumentParser()
p.add_argument("--dataset_root", default=str(Path.home() / "exp/robot/datasets/ac_power_button"))
p.add_argument("--smolvla", default=str(BASE / "smolvla_power_button/checkpoints/007500/pretrained_model"))
p.add_argument("--act", default=str(BASE / "act_power_button/checkpoints/030000/pretrained_model"))
p.add_argument("--episodes", type=int, nargs="+", default=[0, 6, 12, 17])
p.add_argument("--step", type=int, default=5, help="何フレームおきに調べるか")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--go", type=float, default=5.0, help="チャンク内の動きが何度を超えたら「動く」とみなすか")
p.add_argument("--out", default=str(Path.home() / f"exp/robot/logs/switch_{time.strftime('%Y%m%d_%H%M%S')}"))
args = p.parse_args()
dev = "cuda"
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
RENAME_SMOLVLA = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

ds = LeRobotDataset("local/ac_power_button", root=args.dataset_root)
names = [n.removesuffix(".pos") for n in ds.meta.features["action"]["names"]]
J = [i for i, n in enumerate(names) if n != "gripper"]


def load(policy_type, path):
    pol = get_policy_class(policy_type).from_pretrained(path).to(dev)
    pre, post = make_pre_post_processors(
        pol.config,
        pretrained_path=path,
        preprocessor_overrides={
            "device_processor": {"device": dev},
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={"device_processor": {"device": dev}},
    )
    return pol, pre, post


def predict(pol, pre, post, item, rename=None):
    obs = {}
    for k, v in item.items():
        if k.startswith("observation.images."):
            obs[(rename or {}).get(k, k)] = v.unsqueeze(0)
        elif k == "observation.state":
            obs[k] = v
    obs["task"] = item["task"]
    pol.reset()
    with torch.no_grad():
        chunk = pol.predict_action_chunk(pre(obs))
        acts = [post(chunk[:, i, :]) for i in range(chunk.shape[1])]
    return torch.stack(acts, dim=1).squeeze(0).float().cpu().numpy()


def move(a):
    return float(np.abs(a[:, J] - a[0, J]).max())


def to_pil(t):  # (C,H,W) float [0,1] -> 320x240
    arr = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((320, 240))


def save_grid(ep, frames, labels, path):
    W, H, pad = 320, 240, 22
    grid = Image.new("RGB", (W * len(frames), (H + pad) * 2), "white")
    d = ImageDraw.Draw(grid)
    for c, (item, lab) in enumerate(zip(frames, labels, strict=False)):
        for r, key in enumerate(("observation.images.top", "observation.images.wrist")):
            grid.paste(to_pil(item[key]), (c * W, r * (H + pad) + pad))
            d.text((c * W + 4, r * (H + pad) + 4), f"ep{ep} {lab} [{key.split('.')[-1]}]", fill="black")
    grid.save(path)


act = load("act", args.act)
smol = load("smolvla", args.smolvla)
for ep in args.episodes:
    e = ds.meta.episodes[ep]
    lo, hi = int(e["dataset_from_index"]), int(e["dataset_to_index"])
    A = np.stack([ds[j]["action"].numpy() for j in range(lo, hi)])
    dev0 = np.abs(A[:, J] - A[0, J]).max(axis=1)
    onset = int(np.argmax(dev0 > 2.0))
    print(f"=== ep{ep}  動き出し = フレーム{onset}({onset / 30:.1f}秒)")
    print("  フレーム  正解の動き  ACTの動き  SmolVLAの動き")
    sw = {"gt": None, "act": None, "smol": None}
    for k in list(range(0, onset + 1, args.step)):
        item = ds[lo + k]
        g = move(A[k : k + 50])
        a = move(predict(*act, item)[:50])
        sm = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            sm.append(move(predict(*smol, item, RENAME_SMOLVLA)[:50]))
        for name, v in (("gt", g), ("act", a), ("smol", float(np.median(sm)))):
            if sw[name] is None and v > args.go:
                sw[name] = k
        print(f"  {k:6d}  {g:9.1f}  {a:9.1f}  [{' '.join(f'{x:.1f}' for x in sm)}]")
    print(
        f"  切り替わり(動きが{args.go}度を超えた最初のフレーム): 正解={sw['gt']}  ACT={sw['act']}  SmolVLA={sw['smol']}"
    )
    keys = [0] + [v for v in (sw["act"], sw["smol"]) if v is not None] + [max(onset - 30, 0), onset]
    keys = sorted(set(keys))
    labels = []
    for k in keys:
        tag = [t for t, v in (("ACT-go", sw["act"]), ("SmolVLA-go", sw["smol"])) if v == k]
        labels.append(f"f{k}" + (" " + ",".join(tag) if tag else "") + (" onset" if k == onset else ""))
    path = out / f"ep{ep:02d}.png"
    save_grid(ep, [ds[lo + k] for k in keys], labels, path)
    print("  画像:", path)
print("保存先:", out)
