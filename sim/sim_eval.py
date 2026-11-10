"""学習済みの方策(ACT)をシミュの中で動かして評価する。全モデルで同じ場面(乱数の種)を使う。

  python sim_eval.py --models C:\\Users\\yuya\\lerobot\\models --thetas 0 1 2 --episodes 20
  python sim_eval.py --models ... --only act_mixB_sim00 act_sim200       # 一部のモデルだけ
  python sim_eval.py --stub --thetas 1 --episodes 5                      # 方策の代わりにお手本(IK)で動かす(仕組みの確認用)
  python sim_eval.py --models ... --fixed_visual --out sim_eval_vis      # 見た目を較正値に固定して評価(vis0 と比べるとき。出力先を分ける)
  GPU サーバーで並列に(描画は MUJOCO_GL=egl、推論は --device cuda):
    for k in 0 1 2 3; do MUJOCO_GL=egl python sim_eval.py --models M --device cuda --shard $k/4 --out O & done; wait
    python sim_eval.py --summary_only --out O                             # 4つの結果をまとめた表

--models のフォルダの中の、各モデルのフォルダ(pretrained_model の中身: config.json、model.safetensors、
policy_preprocessor.json など)を順に評価する。
結果: sim_eval_results.csv(1エピソード1行)と、モデル × θ の成功率の表(画面と sim_eval_summary.csv)。
--video_n を付けると、各モデル・各 θ の最初の n エピソードを動画(左: 俯瞰、右: 手先)に保存する。

成功: 「運転/停止」が押し込みの80%以上沈み、ほかのボタンは沈んでいない。
失敗の種類: wrong_button(別のボタンを押した)、collision(ボタン以外に当たった、指先が範囲外)、timeout(時間内に押せない)。
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

import gen_sim_dataset as G
import so101_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))


class PolicyRunner:
    """LeRobot の方策(ACT・smolVLA など)を、前処理・後処理込みで1コマずつ動かす(lerobot-rollout の同期推論と同じ手順)。"""

    def __init__(self, path, device="cpu", task="Press the power button"):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors, prepare_observation_for_inference
        self.torch, self.prep = torch, prepare_observation_for_inference
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        self.device = torch.device(device)
        self.task = task
        self.image_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]

    def reset(self):
        self.policy.reset()

    def act(self, images, state):
        obs = {"observation.state": np.asarray(state, np.float32)}
        for k in self.image_keys:
            obs[k] = images[k.split(".")[-1]]  # observation.images.top → "top"
        with self.torch.inference_mode():
            batch = self.prep(obs, self.device, self.task, "so101_follower")
            batch = self.pre(batch)
            action = self.post(self.policy.select_action(batch))
        return action.squeeze(0).cpu().numpy()  # 実機の値(関節の並びはデータセットと同じ)


class StubRunner:
    """方策の代わりに、お手本(IK)の動きを返す。評価の仕組みの確認用。"""

    def __init__(self, sim, ik, units, fps):
        self.sim, self.ik, self.units, self.fps = sim, ik, units, fps
        self.traj, self.k = None, 0

    def reset(self):
        traj, _, _ = G.plan(self.sim, self.ik, np.random.default_rng(0), self.fps, G.MOTION)
        self.traj, self.k = traj, 0

    def act(self, images, state):
        q = self.traj[min(self.k, len(self.traj) - 1)]
        self.k += 1
        return self.units.to_real(q)


def button_joints(sim):
    """パネルの全ボタンの関節(名前 → qpos の番号)。押すボタン以外が沈んだら「別のボタン」。"""
    m = sim.m
    out = {}
    for b in range(m.nbody):
        name = m.body(b).name
        if name.startswith("panel/btn_") and m.body_jntnum[b] > 0:
            out[name.replace("panel/btn_", "")] = m.jnt_qposadr[m.body_jntadr[b]]
    return out


def run_episode(sim, units, runner, home, fps, max_s, buttons, target="power", on_frame=None):
    travel = sim.cfg["button_travel"]
    runner.reset()  # お手本の計画は内部で腕を動かすので、先に済ませてから初期化する
    sim.reset(home)
    res = dict(success=False, t_press=None, reason="timeout", first_hit=None, wrong=None, max_depth_mm=0.0,
               min_dist_mm=1e9)
    import check_alignment as C
    for k in range(int(max_s * fps)):
        imgs = sim.render()
        if on_frame:
            on_frame(imgs)
        a = runner.act(imgs, units.to_real(sim.q()))
        hit = sim.step(units.to_sim(a), 1.0 / fps)
        t = (k + 1) / fps
        if hit and res["first_hit"] is None:
            res["first_hit"] = round(t, 2)
        tip = C.fingertip_world(sim, sim.q())
        res["min_dist_mm"] = min(res["min_dist_mm"], float(np.linalg.norm(tip - sim.button_top()) * 1000))
        res["max_depth_mm"] = max(res["max_depth_mm"], sim.button_depth() * 1000)
        for name, adr in buttons.items():  # 別のボタンが半分以上沈んだ
            if name != target and sim.d.qpos[adr] > 0.5 * travel and res["wrong"] is None:
                res["wrong"] = name
        if sim.button_depth() >= 0.8 * travel and not res["success"] and res["wrong"] is None:
            res["success"], res["t_press"] = True, round(t, 2)
            break
        if res["wrong"] is not None:
            break
    if not res["success"]:
        res["reason"] = "wrong_button" if res["wrong"] else ("collision" if res["first_hit"] is not None else "timeout")
    else:
        res["reason"] = "ok"
    res["min_dist_mm"] = round(res["min_dist_mm"], 1)
    res["max_depth_mm"] = round(res["max_depth_mm"], 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(os.path.dirname(HERE), "models"),
                    help="評価するモデルのフォルダが入ったフォルダ")
    ap.add_argument("--only", nargs="*", default=None, help="この名前を含むモデルだけ評価する")
    ap.add_argument("--stub", action="store_true", help="方策の代わりにお手本(IK)で動かす(仕組みの確認)")
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--episodes", type=int, default=20, help="θ ごとのエピソード数")
    ap.add_argument("--seed", type=int, default=50000, help="場面の乱数の種(学習に使った種と重ならない値)")
    ap.add_argument("--max_s", type=float, default=20.0, help="1エピソードの制限時間 [秒]")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--video_n", type=int, default=0, help="各モデル・各 θ の最初の n エピソードを動画にする")
    ap.add_argument("--shard", default="0/1",
                    help="k/n:エピソード番号を n で割った余りが k のものだけ評価する(n 個のプロセスで並列に回す)")
    ap.add_argument("--task", default="Press the power button", help="言語の指示(smolVLA などで使う)")
    ap.add_argument("--fixed_visual", action="store_true",
                    help="見た目(色・照明・カメラ)を較正した値に固定し、パネルの位置と動きだけ θ でばらつかせる(vis0 と同じ条件)")
    ap.add_argument("--out", default="sim_eval")
    ap.add_argument("--summary_only", action="store_true", help="評価はせず、out の結果をまとめて表にするだけ")
    ap.add_argument("--model_dir", default=os.path.join(os.path.expanduser("~"), "SO-ARM100", "Simulation", "SO101"))
    args = ap.parse_args()
    if args.summary_only:
        summarize(args.out, os.path.join(args.out, "sim_eval_summary.csv"))
        return

    if args.stub:
        models = [("stub_ik", None)]
    else:
        models = sorted((d, os.path.join(args.models, d)) for d in os.listdir(args.models)
                        if os.path.exists(os.path.join(args.models, d, "config.json")))
        if args.only:
            models = [m for m in models if any(o in m[0] for o in args.only)]
    if not models:
        raise SystemExit(f"{args.models} にモデル(config.json のあるフォルダ)がありません")
    print("評価するモデル:", ", ".join(m[0] for m in models))

    os.makedirs(args.out, exist_ok=True)
    k, n_sh = (int(v) for v in args.shard.split("/"))
    suffix = "" if n_sh == 1 else f"_shard{k}of{n_sh}"
    csv_path = os.path.join(args.out, f"sim_eval_results{suffix}.csv")
    done = set()
    if os.path.exists(csv_path):  # 途中から再開できるように、済んだものは飛ばす
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["model"], float(r["theta"]), int(r["episode"])))
    fields = ["model", "theta", "episode", "seed", "success", "reason", "t_press", "first_hit", "wrong",
              "max_depth_mm", "min_dist_mm", "panel_dpos_mm", "panel_drpy_deg", "light_intensity", "wall_s"]
    new_file = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if new_file:
        writer.writeheader()

    for name, path in models:
        sim = S.Sim(args.model_dir)
        units = S.Units(os.path.join(HERE, "units.json"))
        buttons = button_joints(sim)
        runner = StubRunner(sim, S.IK(sim), units, args.fps) if args.stub else PolicyRunner(path, args.device, args.task)
        for th in args.thetas:
            sim.theta = th
            # θ=0 は場面も初期姿勢も毎回同じで、ACT の推論も決定的なので、1回で十分
            n_eps = 1 if th == 0 else args.episodes
            for ep in range(n_eps):
                if ep % n_sh != k:  # 並列に回すとき、ほかのプロセスの受け持ち
                    continue
                if (name, th, ep) in done:
                    continue
                seed = args.seed + int(round(th * 100)) * 1000 + ep  # θ とエピソードで決まる(全モデル共通)
                rng = np.random.default_rng(seed)
                params = sim.randomize(rng)
                if args.fixed_visual:  # 乱数は同じだけ引いたうえで、見た目だけ基準に戻す(gen_sim_dataset.py --fixed_visual と同じ方式)
                    if not hasattr(sim, "restore_visual"):
                        raise SystemExit("so101_sim.py に Sim.restore_visual() がありません(vis0 を作ったときの版に更新してください)")
                    sim.restore_visual()
                home = np.array(G.MOTION["home"]) + rng.normal(0, 1, 6) * np.array(G.MOTION["home_noise"]) * th
                writer_v = None
                if ep < args.video_n:
                    import imageio.v2 as imageio
                    vp = os.path.join(args.out, f"{name}_t{th:g}_ep{ep:02d}.mp4")
                    writer_v = imageio.get_writer(vp, fps=args.fps, macro_block_size=1)
                on_frame = (lambda im: writer_v.append_data(np.concatenate([im["top"], im["wrist"]], 1))) if writer_v else None
                t0 = time.time()
                res = run_episode(sim, units, runner, home, args.fps, args.max_s, buttons, on_frame=on_frame)
                if writer_v:
                    writer_v.close()
                writer.writerow(dict(model=name, theta=th, episode=ep, seed=seed, success=int(res["success"]),
                                     reason=res["reason"], t_press=res["t_press"], first_hit=res["first_hit"],
                                     wrong=res["wrong"], max_depth_mm=res["max_depth_mm"], min_dist_mm=res["min_dist_mm"],
                                     panel_dpos_mm=json.dumps(params["panel_dpos_mm"]),
                                     panel_drpy_deg=json.dumps(params["panel_drpy_deg"]),
                                     light_intensity=params.get("light_intensity"), wall_s=round(time.time() - t0, 1)))
                fcsv.flush()
                print(f"{name} θ={th:g} #{ep:02d}: {res['reason']:12s} 押すまで {res['t_press']}s  "
                      f"一番近づいた {res['min_dist_mm']}mm  ({time.time() - t0:.0f}s)", flush=True)
    fcsv.close()
    summarize(args.out, os.path.join(args.out, "sim_eval_summary.csv"))


def summarize(out_dir, out_path):
    """out_dir の sim_eval_results*.csv(並列に回した分も含む)をまとめて表にする。"""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "sim_eval_results*.csv"))):
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    models = sorted({r["model"] for r in rows})
    thetas = sorted({float(r["theta"]) for r in rows})
    print("\n成功率(成功数 / 回数)と、失敗の内訳(別のボタン / 当たった / 時間切れ)")
    lines = [["model"] + [f"theta{t:g}" for t in thetas] + ["all"]]
    for m in models:
        cells = []
        for t in thetas + [None]:
            rs = [r for r in rows if r["model"] == m and (t is None or float(r["theta"]) == t)]
            ok = sum(int(r["success"]) for r in rs)
            fails = {k: sum(r["reason"] == k for r in rs) for k in ("wrong_button", "collision", "timeout")}
            cells.append(f"{ok}/{len(rs)} ({fails['wrong_button']}/{fails['collision']}/{fails['timeout']})")
        lines.append([m] + cells)
        print(f"  {m:28s} " + "  ".join(f"{c:18s}" for c in cells))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(lines)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()