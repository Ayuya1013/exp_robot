"""MuJoCo 上で IK によるボタン押しのデモを自動生成し、LeRobot のデータセットとして保存する。

使い方(Windows、PowerShell、lerobot の venv で):
  # 1) まず動画で確認(lerobot 不要)
  python gen_sim_dataset.py --model_dir C:\\Users\\yuya\\SO-ARM100\\Simulation\\SO101 --preview preview.mp4 --episodes 5

  # 2) データセットを作る(実機データの info.json から列の定義をそのまま写す)
  python gen_sim_dataset.py --model_dir C:\\Users\\yuya\\SO-ARM100\\Simulation\\SO101 `
      --real_info $env:USERPROFILE\\.cache\\huggingface\\lerobot\\local\\ac_power_button\\meta\\info.json `
      --repo_id local/ac_power_sim --episodes 200

1エピソードの流れ: 待機 → 関節空間でボタンの手前へ → 壁に向かってまっすぐ押す → 少し保持 → 手前へ戻る → 待機姿勢へ戻る。
押し込みが足りない、または手先が壁やパネル(ボタンのまわり以外)にぶつかったエピソードは保存しない。
"""
import argparse
import json
import math
import os
import shutil
import time

import numpy as np

import so101_sim as S

# ===== 動きの設定 =====
MOTION = dict(
    home=[0.0, -1.83, 1.58, 0.97, 0.0, 0.02],  # 待機姿勢 [rad]。check_alignment.py replay の「最初の姿勢」
    home_noise=[0.05, 0.03, 0.03, 0.05, 0.05, 0.02],
    roll_press_deg=(-85.0, 4.0),  # 押すときの手首の回転 [度](平均, 標準偏差)。実機は押す瞬間に約 -82〜-88 度(カメラが爪の上)
    tilt_deg=8.0,                 # 指先の向き: 水平から上下にこの角度以内でランダムに傾ける
    aim_noise=0.003,              # ボタン中心からの狙いのずれ ± [m]
    approach_h=(0.04, 0.08),      # 押す前に止まる位置(ボタン前面から手前へ)
    extra_depth=0.003,            # 押し込みの目標をボタンの底よりこれだけ深くする
    t_start=(0.3, 0.8),           # 最初の静止 [s]
    t_reach=(1.5, 2.5),           # 待機姿勢 → ボタンの手前
    t_down=(0.8, 1.3),            # 壁に向かって押す
    t_hold=(0.2, 0.6),            # 押したまま
    t_up=(0.6, 1.0),              # 手前へ戻る
    t_home=(1.5, 2.5),            # 待機姿勢へ戻る
    t_end=(0.5, 1.0),             # 最後の静止
)
# ======================


def minjerk(n):
    t = np.linspace(0, 1, max(n, 2))
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def plan(sim, ik, rng, fps, mo):
    """1エピソード分の関節目標(フレームごと)を作る。返り値: (関節目標, 待機姿勢, 動きの情報)。作れなければ関節目標が None。"""
    home = np.array(mo["home"]) + rng.uniform(-1, 1, 6) * np.array(mo["home_noise"])
    bt = sim.button_top()
    n = sim.push_dir()  # 壁に向かう向き
    side = np.cross([0, 0, 1.0], n)
    side /= np.linalg.norm(side)
    up = np.cross(n, side)
    a_side, a_up = rng.uniform(-1, 1, 2) * mo["aim_noise"]
    bt += side * a_side + up * a_up
    # 指先の向き: SO-101 は指先の左右の向きを自由に決められない(肩の回転軸から見た方向になる)ので、
    # 回転軸からボタンへの水平な向きを基本に、上下だけランダムに傾ける
    axis = sim.d.xanchor[sim.m.joint("shoulder_pan").id]
    r = np.array([bt[0] - axis[0], bt[1] - axis[1], 0.0])
    r /= np.linalg.norm(r)
    t = math.radians(rng.uniform(-mo["tilt_deg"], mo["tilt_deg"]))
    d = math.cos(t) * r + math.sin(t) * np.array([0, 0, 1.0])
    depth = sim.cfg["button_travel"] + mo["extra_depth"]
    h_app = rng.uniform(*mo["approach_h"])
    p_above = bt - d * h_app
    p_press = bt + d * depth

    seed = home.copy()
    m_, s_ = mo["roll_press_deg"]  # 手首の回転はエピソードごとに決めて固定(IK は残りの関節で位置を合わせる)
    seed[4] = math.radians(m_ + s_ * float(np.clip(rng.normal(), -2.5, 2.5)))
    q_above, ep, ea = ik.solve(seed, p_above, d)
    if ep > 0.002 or ea > math.radians(6):  # 手首を回すと指先の向きが数度ずれる(指先は回転軸から約8mm離れている)
        return None, home, None
    nf = lambda key: max(2, int(round(rng.uniform(*mo[key]) * fps)))
    traj = [home] * nf("t_start")

    s = minjerk(nf("t_reach"))  # 関節空間でボタンの手前へ
    traj += [home + (q_above - home) * si for si in s]

    q = q_above
    for key, a, b in (("t_down", p_above, p_press), ("t_hold", p_press, p_press), ("t_up", p_press, p_above)):
        for si in minjerk(nf(key)):  # 手先をまっすぐ動かす
            q, ep, ea = ik.solve(q, a + (b - a) * si, d, iters=50)
            if ep > 0.004:
                return None, home, None
            traj.append(q.copy())
        if key == "t_down":
            i_press = len(traj) - 1

    s = minjerk(nf("t_home"))
    traj += [q + (home - q) * si for si in s]
    traj += [home] * nf("t_end")
    info = dict(tilt_deg=round(math.degrees(t), 2), roll_deg=round(math.degrees(seed[4]), 1),
                approach_mm=round(h_app * 1000, 1), aim_mm=[round(a_side * 1000, 2), round(a_up * 1000, 2)],
                press_frame=i_press, frames=len(traj))
    return np.array(traj), home, info


def run(sim, traj, home, fps, on_frame=None):
    """traj を実行する。on_frame があれば各フレームで (画像, 状態, 目標) を渡す。返り値: (押し込み量の最大, 衝突したか)"""
    sim.reset(home)
    max_depth, hit = 0.0, False
    for target in traj:
        if on_frame is not None:
            on_frame(sim.render(), sim.q(), target)
        hit |= sim.step(target, 1.0 / fps)
        max_depth = max(max_depth, sim.button_depth())
        if max_depth >= 0.7 * sim.cfg["button_travel"]:
            sim.set_lcd(True)  # 押したら画面が光る(実機と同じ)
    return max_depth, hit


def load_features(info_path, cfg):
    """実機データセットの info.json から列の定義を読む。画像サイズも合わせる。"""
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    feats = {}
    for k, v in info["features"].items():
        if k in skip:
            continue
        v = {kk: vv for kk, vv in v.items() if kk != "info"}
        v["shape"] = tuple(v["shape"])
        feats[k] = v
    for k, v in feats.items():
        if k.startswith("observation.images."):
            h, w = v["shape"][0], v["shape"][1]
            if (h, w) != (cfg["img_h"], cfg["img_w"]):
                print(f"画像サイズを実機に合わせます: {k} {w}x{h}")
                cfg["img_h"], cfg["img_w"] = h, w
    return feats, info.get("fps", 30), info.get("robot_type", "so101_follower")


def default_features(cfg):
    names = [f"{j}.pos" for j in S.JOINTS]
    feats = {
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
    }
    for cam in ("wrist", "top"):
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (cfg["img_h"], cfg["img_w"], 3),
                                              "names": ["height", "width", "channels"]}
    return feats


def joint_order(names):
    """データセットの列の並び(例: "shoulder_pan.pos")を JOINTS の番号に直す。"""
    return [S.JOINTS.index(n.removesuffix(".pos")) for n in names]


def make_sheet(sim, ik, rng, fps, args):
    """ランダム化の様子を確かめる一覧画像。成功するエピソードの「押す瞬間」の俯瞰と手先の画像を並べる。"""
    import cv2
    import imageio.v2 as imageio
    tiles, tried = [], 0
    while len(tiles) < args.sheet_n and tried < args.sheet_n * 5:
        tried += 1
        params = sim.randomize(rng, visual=not args.no_random_visual)
        if args.fixed_visual:
            sim.restore_visual()
        traj, home, info = plan(sim, ik, rng, fps, MOTION)
        if traj is None:
            continue
        frames = {}
        sim.reset(home)  # 画像は押す瞬間だけ描く(速さのため)
        depth, hit = 0.0, False
        for k, target in enumerate(traj):
            if k == info["press_frame"]:
                frames["press"] = sim.render()
            hit |= sim.step(target, 1.0 / fps)
            depth = max(depth, sim.button_depth())
        im = np.concatenate([frames["press"]["top"], frames["press"]["wrist"]], axis=1)
        im = cv2.resize(im, (640, 240))
        ok = "ok" if (not hit and depth >= 0.8 * sim.cfg["button_travel"]) else ("hit" if hit else "shallow")
        cv2.putText(im, f"#{len(tiles)} {ok} light {params.get('light_intensity', 1):.2f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        tiles.append(im)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    imageio.imwrite(args.sheet, np.concatenate(rows, axis=0))
    print(f"一覧: {os.path.abspath(args.sheet)}(各マス 左: 俯瞰、右: 手先。押す瞬間)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="SO-ARM100/Simulation/SO101 のフォルダ")
    ap.add_argument("--episodes", type=int, default=50, help="保存する(成功した)エピソード数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", default="", help="指定すると LeRobot には保存せず、この mp4 に動画を書く")
    ap.add_argument("--real_info", default="", help="実機データセットの meta/info.json")
    ap.add_argument("--repo_id", default="local/ac_power_sim")
    ap.add_argument("--root", default="", help="保存先フォルダ(省略時は ~/.cache/huggingface/lerobot/<repo_id>)")
    ap.add_argument("--task", default="Press the power button")
    ap.add_argument("--units", default="units.json")
    ap.add_argument("--overwrite", action="store_true", help="保存先があれば消して作り直す")
    ap.add_argument("--no_random_visual", action="store_true", help="色・照明・カメラのランダム化をしない")
    ap.add_argument("--fixed_visual", action="store_true",
                    help="見た目とカメラを較正した基準の値に固定する。乱数は --fixed_visual なしと同じだけ引くので、"
                         "同じ --seed と --theta なら、パネルの位置・動き・保存されるエピソードがなしの場合と1本ずつ一致する")
    ap.add_argument("--theta", type=float, default=1.0,
                    help="ランダム化の強さ。so101_sim.py の RANDOM(θ=1 の標準偏差)を θ 倍する。0 でばらつきなし")
    ap.add_argument("--sheet", default="", help="指定すると、押す瞬間の画像を並べた一覧(png)だけを作って終わる")
    ap.add_argument("--sheet_n", type=int, default=12, help="--sheet で並べるエピソード数")
    args = ap.parse_args()

    cfg = dict(S.CFG)
    if args.real_info:
        feats, fps, robot_type = load_features(args.real_info, cfg)
    else:
        feats, fps, robot_type = default_features(cfg), 30, "so101_follower"
    sim = S.Sim(args.model_dir, cfg, theta=args.theta)
    ik = S.IK(sim)
    units = S.Units(args.units)
    print(f"単位変換: {units.source}")
    order = joint_order(feats["action"]["names"])
    cams = {k: k.removeprefix("observation.images.") for k in feats if k.startswith("observation.images.")}
    for cam in cams.values():
        if cam not in ("top", "wrist"):
            raise SystemExit(f"カメラ名 {cam} はシミュレーションにありません(top と wrist のみ)。")
    rng = np.random.default_rng(args.seed)
    print(f"ランダム化の強さ θ = {args.theta}  乱数の種 {args.seed}"
          + ("  見た目とカメラは基準に固定(--fixed_visual)" if args.fixed_visual else ""))
    if args.fixed_visual and args.no_random_visual:
        raise SystemExit("--fixed_visual と --no_random_visual は一緒に使えません(乱数の引き方が変わります)。")

    if args.sheet:
        make_sheet(sim, ik, rng, fps, args)
        return

    writer, ds, log = None, None, None
    if args.preview:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.preview, fps=fps, macro_block_size=8)
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = args.root or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot", args.repo_id)
        if os.path.exists(root):
            if not args.overwrite:
                raise SystemExit(f"保存先が既にあります: {root}(作り直すなら --overwrite)")
            shutil.rmtree(root)
        ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=feats, root=root,
                                   robot_type=robot_type, use_videos=True)
        print(f"保存先: {root}")
        # エピソードごとのランダム化と動きの値を、データセットの隣に記録する(データセットの中には入れない)
        log_path = root.rstrip("/\\") + "_sim_params.jsonl"
        log = open(log_path, "w", encoding="utf-8")
        print(f"ランダム化の記録: {log_path}")

    saved, tried, t0 = 0, 0, time.time()
    fails = {"IK": 0, "衝突": 0, "押し不足": 0}
    try:
        while saved < args.episodes:
            tried += 1
            params = sim.randomize(rng, visual=not args.no_random_visual)
            if args.fixed_visual:  # 乱数は引いたまま、見た目とカメラだけ基準に戻す
                sim.restore_visual()
                params["visual_applied"] = False
            traj, home, info = plan(sim, ik, rng, fps, MOTION)
            if traj is None:
                fails["IK"] += 1
                continue
            depth, hit = run(sim, traj, home, fps)  # まず画像なしで試す
            if hit or depth < 0.8 * sim.cfg["button_travel"]:
                fails["衝突" if hit else "押し不足"] += 1
                continue

            def on_frame(imgs, q, target):  # 同じ条件で、今度は記録しながら動かす(結果は同じになる)
                if writer is not None:
                    writer.append_data(np.concatenate([imgs["top"], imgs["wrist"]], axis=1)[::2, ::2])
                    return
                frame = {"observation.state": units.to_real(q)[order], "action": units.to_real(target)[order],
                         "task": args.task}
                for key, cam in cams.items():
                    frame[key] = imgs[cam]
                ds.add_frame(frame)

            run(sim, traj, home, fps, on_frame)
            if ds is not None:
                ds.save_episode()
                log.write(json.dumps(dict(episode=saved, seed=args.seed, depth_mm=round(depth * 1000, 2),
                                          **params, motion=info), ensure_ascii=False) + "\n")
                log.flush()
            saved += 1
            print(f"[{saved}/{args.episodes}] {len(traj)} フレーム  押し込み {depth * 1000:.1f}mm  "
                  f"試行 {tried}  失敗 {fails}  経過 {time.time() - t0:.0f}s")
    finally:
        if writer is not None:
            writer.close()
            print(f"動画: {os.path.abspath(args.preview)}")
        if ds is not None:
            ds.finalize()
        if log is not None:
            log.close()


if __name__ == "__main__":
    main()