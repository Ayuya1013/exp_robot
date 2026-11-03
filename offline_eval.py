"""データセットの各エピソードの「最初のフレーム」と「最後のフレーム」を方策に入れ、
出てくるチャンクが動き出すか(どれだけ動くか)を、正解(デモの次の動作)と比べる。

LeRobot 0.6.2(eb7a54c6)の policy_server と同じ手順で推論する:
  from_pretrained → make_pre_post_processors → preprocessor → predict_action_chunk → postprocessor(1動作ずつ)

使い方(研究室Linuxサーバー):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python offline_eval.py
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

BASE = Path.home() / "exp/robot/lerobot/outputs/train"
p = argparse.ArgumentParser()
p.add_argument("--dataset_root", default=str(Path.home() / "exp/robot/datasets/ac_power_button"))
p.add_argument("--smolvla", default=str(BASE / "smolvla_power_button/checkpoints/007500/pretrained_model"))
p.add_argument("--act", default=str(BASE / "act_power_button/checkpoints/030000/pretrained_model"))
p.add_argument("--seeds", type=int, default=5, help="SmolVLAのノイズを変えて何回生成するか")
p.add_argument("--out", default=str(Path.home() / f"exp/robot/logs/offline_eval_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
args = p.parse_args()
dev = "cuda"
RENAME_SMOLVLA = {"observation.images.top": "observation.images.camera1",
                  "observation.images.wrist": "observation.images.camera2"}

ds = LeRobotDataset("local/ac_power_button", root=args.dataset_root)
names = [n.removesuffix(".pos") for n in ds.meta.features["action"]["names"]]
J_MOVE = [names.index("shoulder_lift"), names.index("elbow_flex")]  # 腕を伸ばすときに大きく動く関節
print("関節:", names)


def load(policy_type, path):
    pol = get_policy_class(policy_type).from_pretrained(path).to(dev)
    pre, post = make_pre_post_processors(
        pol.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": dev},
                                "rename_observations_processor": {"rename_map": {}}},
        postprocessor_overrides={"device_processor": {"device": dev}},
    )
    return pol, pre, post


def predict(pol, pre, post, item, rename=None):
    obs = {}
    for k, v in item.items():
        if k.startswith("observation.images."):
            obs[(rename or {}).get(k, k)] = v.unsqueeze(0)  # サーバーと同じく画像だけバッチ次元を付ける
        elif k == "observation.state":
            obs[k] = v
    obs["task"] = item["task"]
    pol.reset()
    with torch.no_grad():
        chunk = pol.predict_action_chunk(pre(obs))  # (1, T, D)
        acts = [post(chunk[:, i, :]) for i in range(chunk.shape[1])]
    return torch.stack(acts, dim=1).squeeze(0).float().cpu().numpy()  # (T, D)


def move(a):  # チャンク内で、腕の関節が最初からどれだけ動いたか(度、最大値)
    return float(np.abs(a[:, J_MOVE] - a[0, J_MOVE]).max())


models = {"smolvla": load("smolvla", args.smolvla), "act": load("act", args.act)}
rows = []
for ep in range(ds.meta.total_episodes):
    e = ds.meta.episodes[ep]
    lo, hi = int(e["dataset_from_index"]), int(e["dataset_to_index"])
    for where, idx in (("start", lo), ("end", hi - 1)):
        item = ds[idx]
        st = item["observation.state"].numpy()
        gt = np.stack([ds[j]["action"].numpy() for j in range(idx, min(idx + 50, hi))])
        r = {"ep": ep, "where": where, "task": item["task"],
             **{f"state_{n}": round(float(st[i]), 2) for i, n in enumerate(names)},
             "gt_move50": round(move(gt), 2)}
        a = predict(*models["act"], item)
        r["act_move50"] = round(move(a[:50]), 2)
        sm = []
        for s in range(args.seeds):
            torch.manual_seed(s)
            sm.append(move(predict(*models["smolvla"], item, RENAME_SMOLVLA)))
        r["smolvla_moves"] = " ".join(f"{m:.1f}" for m in sm)
        r["smolvla_move_mean"] = round(float(np.mean(sm)), 2)
        rows.append(r)
        print(f"ep{ep:2d} {where:5s} 正解{r['gt_move50']:6.1f}  ACT{r['act_move50']:6.1f}  "
              f"SmolVLA[{r['smolvla_moves']}]  手首(flex,roll)=({st[names.index('wrist_flex')]:.0f},{st[names.index('wrist_roll')]:.0f})")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("保存:", args.out)
