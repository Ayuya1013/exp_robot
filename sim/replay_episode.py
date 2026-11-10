"""実機で録ったエピソード(LeRobot v3 データセット)を、シミュのシーンで再生する。LeRobot は不要(pandas と pyarrow だけ)。

使い方:
    python replay_episode.py --root ~/datasets/ac_power_button --episode 3                 # ビューアで再生
    python replay_episode.py --root ... --episode 3 --mode kinematic                       # 記録した姿勢をそのまま置く
    python replay_episode.py --root ... --episode 3 --video ep3.mp4                        # 動画にする(画面なしでも可)
    python replay_episode.py --root ... --all                                              # 全エピソードを集計だけする

モード:
    physics   : 記録した action(リーダーの角度)を目標にして物理シミュで動かす。壁やボタンに当たれば止まる。
                「シミュのボタンを実際に押せるか」を確かめる(既定)
    kinematic : 記録した observation.state(フォロワーの実際の角度)をそのまま置く。壁にめり込んでも止まらない。
                「シミュのパネルの位置が実機と合っているか」を確かめる
最後に、押した瞬間の指先(gripperframe)が、押す位置(press_target)からどれだけずれていたかを、パネルの向きで表示する。
データセットは data/ と meta/ だけあればよい(videos/ は不要)。units.json があれば実機の値の変換に使う。
"""
import argparse
import glob
import json
import math
import os
import time

import mujoco
import numpy as np
import pandas as pd

import so101_sim as S


def load_episodes(root):
    with open(os.path.join(root, "meta", "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"{root}/data に parquet がありません")
    cols = ["episode_index", "frame_index", "observation.state", "action"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    names = info["features"]["action"]["names"]
    order = [S.JOINTS.index(n.removesuffix(".pos")) for n in names]
    return df, order, info.get("fps", 30)


def episode_arrays(df, ep, order):
    e = df[df["episode_index"] == ep].sort_values("frame_index")
    if e.empty:
        raise SystemExit(f"エピソード {ep} がありません(0〜{df['episode_index'].max()})")
    st, ac = np.zeros((len(e), 6)), np.zeros((len(e), 6))
    st[:, order] = np.stack(e["observation.state"].to_numpy())
    ac[:, order] = np.stack(e["action"].to_numpy())
    return st, ac


def press_offset(sim, units, states):
    """指先が一番前(押す向き)に出たフレームで、指先が press_target からどれだけずれていたか。
    返り値: (フレーム, [右, 上, 奥] mm)。右・上はパネルを正面から見た向き、奥は押し込む向き。"""
    ik = S.IK(sim)
    mujoco.mj_forward(sim.m, sim.d)
    R = sim.d.xmat[sim.mount].reshape(3, 3)  # 列: パネルの x(右), y(奥), z(上)
    tgt = sim.button_top()
    best = None
    for k, st in enumerate(states):
        p, _ = ik.fk(units.to_sim(st))
        rel = R.T @ (p - tgt)
        if best is None or rel[1] > best[1][1]:
            best = (k, rel)
    k, rel = best
    return k, np.array([rel[0], rel[2], rel[1]]) * 1000


def run_episode(sim, units, st, ac, fps, mode, on_frame=None):
    """1エピソード再生する。返り値: 集計(dict)"""
    m, d = sim.m, sim.d
    sim.reset(units.to_sim(st[0]))
    max_depth, first_press, hit_frames, max_pen = 0.0, None, 0, 0.0
    travel = sim.cfg["button_travel"]
    for k in range(len(st)):
        if mode == "physics":
            hit = sim.step(units.to_sim(ac[k]), 1.0 / fps)
        else:
            d.qpos[sim.qadr] = units.to_sim(st[k])
            mujoco.mj_forward(m, d)
            hit = False
            top = sim.button_top()
            for c in d.contact[:d.ncon]:  # ボタンから3cm以上離れた所で、腕がパネルや壁にめり込んだ深さ
                pair = {c.geom1, c.geom2}
                if pair & sim.dev_geoms and pair & sim.robot_geoms and np.linalg.norm(c.pos - top) > 0.03:
                    max_pen = max(max_pen, -c.dist)
                    hit = True
        hit_frames += int(hit)
        depth = sim.button_depth()
        if depth > max_depth:
            max_depth = depth
        if first_press is None and depth >= 0.7 * travel:
            first_press = k / fps
            sim.set_lcd(True)
        if on_frame is not None and on_frame(k) is False:
            break
    return dict(max_depth_mm=max_depth * 1000, first_press_s=first_press, hit_frames=hit_frames,
                max_pen_mm=max_pen * 1000)


def report(ep, res, off, fps, mode):
    line = f"エピソード {ep:3d}: "
    if mode == "physics":
        pressed = f"押せた({res['first_press_s']:.1f}秒)" if res["first_press_s"] is not None else "押せなかった"
        line += f"{pressed}  押し込み最大 {res['max_depth_mm']:.1f}mm  ほかの所に当たったフレーム {res['hit_frames']}"
    else:  # 姿勢を置くだけなのでボタンは動かない。ボタン以外へのめり込みを見る
        line += f"ボタン以外にめり込んだフレーム {res['hit_frames']}(最大 {res['max_pen_mm']:.1f}mm)"
    k, o = off
    line += f"  | 押した瞬間({k / fps:.1f}秒)の指先のずれ 右{o[0]:+.0f} 上{o[1]:+.0f} 奥{o[2]:+.0f} mm"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="データセットのフォルダ(meta/ と data/ がある所)")
    ap.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="全エピソードを再生して集計だけ出す(画面なし)")
    ap.add_argument("--mode", choices=["physics", "kinematic"], default="physics")
    ap.add_argument("--speed", type=float, default=1.0, help="ビューアでの再生速度")
    ap.add_argument("--loop", action="store_true", help="ビューアでくり返し再生する")
    ap.add_argument("--video", default="", help="指定すると、ビューアを開かずにこの mp4 を作る")
    ap.add_argument("--units", default=os.path.join(S.HERE, "units.json"))
    args = ap.parse_args()

    df, order, fps = load_episodes(args.root)
    units = S.Units(args.units)
    print(f"単位変換: {units.source}  /  エピソード数 {df['episode_index'].nunique()}  /  {fps}fps  /  モード {args.mode}")
    sim = S.Sim(args.model_dir, render=bool(args.video))

    # シミュで作ったデータなら、生成のときのパネルの位置のずれ(_sim_params.jsonl)を再現する
    sim_params = {}
    pp = args.root.rstrip("/\\") + "_sim_params.jsonl"
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sim_params[r["episode"]] = r
        print(f"シミュのデータ: 生成のときのパネルの位置を再現します({pp})")
    base = sim.env["panel"]

    def set_scene(ep):
        import env_builder as E
        r = sim_params.get(int(ep))
        if r is None:
            E.set_panel_pose(sim.m, base["panel_pos"], base["panel_rpy_deg"])
        else:
            E.set_panel_pose(sim.m, np.array(base["panel_pos"]) + np.array(r["panel_dpos_mm"]) / 1000,
                             np.array(base["panel_rpy_deg"]) + np.array(r["panel_drpy_deg"]))
        mujoco.mj_forward(sim.m, sim.d)

    if args.all:
        offs = []
        for ep in sorted(df["episode_index"].unique()):
            set_scene(ep)
            st, ac = episode_arrays(df, ep, order)
            res = run_episode(sim, units, st, ac, fps, args.mode)
            off = press_offset(sim, units, st)
            offs.append(off[1])
            report(ep, res, off, fps, args.mode)
        o = np.mean(offs, axis=0)
        print(f"\n指先のずれの平均: 右{o[0]:+.0f} 上{o[1]:+.0f} 奥{o[2]:+.0f} mm(標準偏差 右{np.std(offs, 0)[0]:.0f} 上{np.std(offs, 0)[1]:.0f} mm)")
        print("右と上の平均が大きく 0 からずれていれば、パネルの位置がずれている可能性がある"
              "(指先の目印 gripperframe と、実際にボタンに当たる所は少し違うので、数mmの差は気にしなくてよい)。")
        return

    set_scene(args.episode)
    st, ac = episode_arrays(df, args.episode, order)
    print(f"エピソード {args.episode}: {len(st)} フレーム({len(st) / fps:.1f}秒)")

    if args.video:
        import imageio.v2 as imageio
        side = mujoco.Renderer(sim.m, sim.cfg["img_h"], sim.cfg["img_w"])
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.16, -0.04, 0.17], 0.85, 90, -5
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0  # 横からの画像では壁を隠す
        w = imageio.get_writer(args.video, fps=fps, macro_block_size=8)

        def on_frame(k):
            im = sim.render()
            side.update_scene(sim.d, cam, scene_option=opt)
            frame = np.concatenate([side.render(), im["top"], im["wrist"]], axis=1)
            w.append_data(frame[::2, ::2])

        res = run_episode(sim, units, st, ac, fps, args.mode, on_frame)
        w.close()
        print(f"動画: {os.path.abspath(args.video)}(左: 横から、中: top、右: wrist)")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(sim.m, sim.d) as v:
            while True:
                t0 = time.time()

                def on_frame(k):
                    v.sync()
                    time.sleep(max(0.0, (k + 1) / (fps * args.speed) - (time.time() - t0)))
                    return v.is_running()

                res = run_episode(sim, units, st, ac, fps, args.mode, on_frame)
                if not (args.loop and v.is_running()):
                    break
            report(args.episode, res, press_offset(sim, units, st), fps, args.mode)
            print("ビューアを閉じると終了します。")
            while v.is_running():
                v.sync()
                time.sleep(0.05)
        return
    report(args.episode, res, press_offset(sim, units, st), fps, args.mode)


if __name__ == "__main__":
    main()