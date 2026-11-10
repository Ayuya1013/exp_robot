"""段階1: 実機とシミュレーションの関節の対応(向き・ずれ)を確かめて units.json に保存する。

■ live: 実機の腕を手で動かすと、シミュレーションの腕が同じように動くかをその場で見る(おすすめ)
    python check_alignment.py live --model_dir C:\\Users\\yuya\\SO-ARM100\\Simulation\\SO101 --port COM4 --id my_follower
  起動するとモーターの力が抜けるので、腕を手で支えてから始めること。
  コンソールで入力するコマンド:
    Enter だけ      … 今の値を表示(実機の値と、シミュレーションでの角度)
    zero            … 入力して3秒以内に、腕を基準の姿勢(上腕を真上に立て、前腕と手先を水平に前へ伸ばした L 字。
                      zero_pose.png と同じ形)にして支える。5関節のずれを記録
    flip 関節名      … その関節の向きを反対にする(例: flip elbow_flex)
    gclose / gopen  … 3秒後に、グリッパーを完全に閉じた/開いた状態を記録
    save            … units.json に保存     q … 終了(保存はしない)

■ replay: 録画済みの実機データの1エピソードを、シミュレーションで同じ姿勢にして動画にする
    python check_alignment.py replay --model_dir ... --repo_id local/ac_power_button --episode 0
    python check_alignment.py replay --model_dir ... --episode 0,1,2,3,4,5,6,7,8,9 --no_video   … 押した位置だけ推定
  出力: replay_ep0.mp4(左上: 実機の top、右上: シミュレーションの top、左下: 2つを重ねたもの、右下: 横から見たシミュレーション)
  あわせて、最初の姿勢(gen_sim_dataset.py の home に使う値)と、関節が可動範囲を超えていないかを表示する。
"""
import argparse
import json
import math
import os
import re
import threading
import time

import mujoco
import numpy as np

import env_builder as E
import so101_sim as S

SIM_GRIPPER_CLOSED, SIM_GRIPPER_OPEN = -0.17453, 1.74533


def save_units(units, path):
    data = {j: {"scale": round(float(units.scale[i]), 5), "offset": round(float(units.offset[i]), 3)}
            for i, j in enumerate(S.JOINTS)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"保存しました: {os.path.abspath(path)}")


def side_camera():
    """横(ロボットの右側)から見るカメラ。壁は side_option() で隠す。"""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.18, 0.0, 0.15]
    cam.distance, cam.azimuth, cam.elevation = 0.8, 90, -10
    return cam


def side_option():
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0  # 壁(グループ1)を描かない
    return opt


def save_zero_pose_png(sim, path="zero_pose.png"):
    """シミュレーションの q=0(各関節が可動範囲の中央)の姿勢を、横と正面から描いて保存する。"""
    import imageio.v2 as imageio
    sim.reset(np.zeros(6))
    r = mujoco.Renderer(sim.m, 360, 480)
    imgs = []
    for az in (90, 180):
        cam = side_camera()
        cam.azimuth, cam.elevation = az, -5
        r.update_scene(sim.d, cam, scene_option=side_option())
        imgs.append(r.render())
    imageio.imwrite(path, np.concatenate(imgs, axis=1))
    r.close()
    print(f"基準の姿勢の画像: {os.path.abspath(path)}(左: 横から、右: 正面から)")


# ---------------- live ----------------
HELP = ("コマンド(入力して Enter):\n"
        "  Enter だけ     … 今の値を表示\n"
        "  zero           … 3秒後に基準の姿勢(L字)を記録。入力したらすぐ両手で腕を支えて形を作る\n"
        "  flip 関節名     … その関節の向きを反対にする(例: flip elbow_flex)\n"
        "  gclose / gopen … 3秒後にグリッパーの閉/開を記録\n"
        "  --- パネルの位置合わせ(固定側の爪の先を使う) ---\n"
        "  btn            … 3秒後、爪の先を当てている所を「運転/停止」の前面の中心として、パネルを置き直す\n"
        "  move dx dy dz  … パネルを動かす [mm](前、左、上)。例: move 0 0 -3\n"
        "  wall           … 3秒後、爪の先を当てている壁の位置を測り、ボタンの面からの奥行きを表示する\n"
        "  save           … units.json と panel.json に保存     q … 終了")


def fingertip_local(m):
    """固定側の爪の先端(gripper ボディの座標系)。当たり判定用メッシュで、指先の向き(-z)に一番出ている点。"""
    bid = m.body("gripper").id
    best = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] == 0:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        p = v @ R.reshape(3, 3).T + m.geom_pos[g]
        i = np.argmin(p[:, 2])
        if best is None or p[i, 2] < best[2]:
            best = p[i]
    return best


def fingertip_world(sim, q):
    """関節角 q [rad] のときの、固定側の爪の先端の位置。"""
    d = mujoco.MjData(sim.m)
    d.qpos[sim.qadr] = q
    mujoco.mj_kinematics(sim.m, d)
    g = sim.m.body("gripper").id
    return d.xpos[g] + d.xmat[g].reshape(3, 3) @ fingertip_local(sim.m)


def panel_offset_mm(sim, tip):
    """爪の先が、押す位置(press_target)からどれだけずれているか [右, 上, 奥] mm(パネルを正面から見た向き)。"""
    mujoco.mj_forward(sim.m, sim.d)
    R = sim.d.xmat[sim.mount].reshape(3, 3)
    rel = R.T @ (tip - sim.button_top())
    return np.array([rel[0], rel[2], rel[1]]) * 1000


def status_line(sim, units, real):
    q = units.to_sim(real)
    lo, hi = sim.m.jnt_range[[sim.m.joint(j).id for j in S.JOINTS]].T
    lines = []
    for i, j in enumerate(S.JOINTS):
        over = "  ※可動範囲外" if (q[i] < lo[i] - 0.05 or q[i] > hi[i] + 0.05) else ""
        lines.append(f"  {j:14s} 実機 {real[i]:+7.1f}  →  シミュレーション {math.degrees(q[i]):+6.1f}度"
                     f"  (範囲 {math.degrees(lo[i]):+.0f}〜{math.degrees(hi[i]):+.0f}){over}")
    # 固定側の爪の先端の位置(順運動学)。定規で測った値と比べる
    tip = fingertip_world(sim, q)
    axis = sim.d.xanchor[sim.m.joint("shoulder_pan").id]
    rel = (tip - axis) * 100
    lines.append(f"  爪の先端: 肩の回転軸から 前 {rel[0]:+.1f}  左 {rel[1]:+.1f} cm、台の底面から 高さ {tip[2] * 100:+.1f} cm")
    o = panel_offset_mm(sim, tip)
    lines.append(f"  「運転/停止」の前面の中心からのずれ: 右 {o[0]:+.0f}  上 {o[1]:+.0f}  奥 {o[2]:+.0f} mm")
    return "\n".join(lines)


def averaged(get_real, wait_s=3.0, n=10):
    """wait_s 秒数えてから、0.5秒間の平均をとる(手で支えている間の揺れを減らす)。"""
    for k in range(int(wait_s), 0, -1):
        print(f"  {k}...", flush=True)
        time.sleep(1.0)
    samples = []
    for _ in range(n):
        samples.append(get_real())
        time.sleep(0.05)
    return np.mean(samples, axis=0)


def handle_command(cmd, units, get_real, path, sim=None, wait_s=3.0):
    """コンソールのコマンドを処理する。get_real() は今の実機の値(JOINTS の順)を返す。終了なら False。"""
    parts = cmd.strip().split()
    if not parts:
        if sim is not None:
            print(status_line(sim, units, get_real()))
        return True
    c = re.sub(r"[^a-z_]", "", parts[0].lower())  # 打ち間違いの記号は無視
    if c == "zero":
        real = averaged(get_real, wait_s)
        for i in range(5):  # この姿勢がシミュレーションの 0 ラジアン
            units.offset[i] = real[i]
        print("ずれを記録: " + ", ".join(f"{j}={real[i]:+.1f}" for i, j in enumerate(S.JOINTS[:5])))
        print("(まだ保存していません。確認できたら save)")
    elif c == "flip" and len(parts) == 2 and parts[1] in S.JOINTS:
        i = S.JOINTS.index(parts[1])
        units.scale[i] *= -1
        print(f"{parts[1]} の向きを反対にしました(scale={units.scale[i]:+.3f})")
    elif c in ("gclose", "gopen"):
        real = averaged(get_real, wait_s)
        units._g = getattr(units, "_g", {})
        units._g[c] = real[5]
        print(f"グリッパー {c}: {real[5]:.1f}")
        if len(units._g) == 2:
            a, b = units._g["gclose"], units._g["gopen"]
            units.scale[5] = (b - a) / (SIM_GRIPPER_OPEN - SIM_GRIPPER_CLOSED)
            units.offset[5] = a - units.scale[5] * SIM_GRIPPER_CLOSED
            print(f"グリッパーの対応: scale={units.scale[5]:.3f}, offset={units.offset[5]:.2f}")
    elif c in ("btn", "move", "wall") and sim is not None:
        handle_panel_command(c, parts, units, get_real, sim, wait_s)
    elif c == "save":
        save_units(units, path)
        if sim is not None and getattr(sim, "_panel_changed", False):
            p = sim.env["panel"]
            out = os.path.join(E.HERE, "panel.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"button_pos": [round(float(v), 4) for v in p["button_pos"]], "panel_pos": None}, f, indent=2)
            print(f"保存しました: {os.path.abspath(out)}(env_builder.py がこの値を使う)")
    elif c == "q":
        return False
    else:
        print(f"「{cmd.strip()}」は分かりませんでした。\n" + HELP)
    return True


def handle_panel_command(c, parts, units, get_real, sim, wait_s):
    p = sim.env["panel"]
    if c == "wall":
        tip = fingertip_world(sim, units.to_sim(averaged(get_real, wait_s)))
        depth = (tip - sim.button_top()) @ sim.push_dir() * 1000
        board = sim.m.geom("panel/panel_board")
        thick = 2 * sim.m.geom_size[board.id][1] * 1000
        print(f"壁は「運転/停止」の前面から {depth:+.0f} mm 奥(xml のパネルの厚みは {thick:.0f} mm)")
        return
    if c == "btn":
        tip = fingertip_world(sim, units.to_sim(averaged(get_real, wait_s)))
        p["button_pos"] = tip.tolist()
    else:  # move
        try:
            dx, dy, dz = (float(v) / 1000 for v in parts[1:4])
        except ValueError:
            print("使い方: move 前 左 上 [mm](例: move 0 0 -3)")
            return
        p["button_pos"] = (np.array(p["button_pos"]) + [dx, dy, dz]).tolist()
    p["panel_pos"] = E.panel_pos_from_button(p["button_pos"], p["panel_rpy_deg"], p["target_button"],
                                             p["panel_xml"]).tolist()
    E.set_panel_pose(sim.m, p["panel_pos"], p["panel_rpy_deg"])
    mujoco.mj_forward(sim.m, sim.d)
    sim._panel_changed = True
    b = np.array(p["button_pos"]) * 100
    print(f"パネルを置き直しました。「運転/停止」の前面: x {b[0]:.1f}  y {b[1]:.1f}  z {b[2]:.1f} cm(save で保存)")


def live(args):
    import mujoco.viewer
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    sim = S.Sim(args.model_dir, render=False)
    save_zero_pose_png(sim)
    units = S.Units(args.units)
    print(f"単位変換の初期値: {units.source}")

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect(calibrate=False)
    robot.bus.disable_torque()
    print("モーターの力を抜きました。腕を手で動かしてください。")
    print(HELP)

    state = {"real": np.zeros(6), "run": True}
    get_real = lambda: state["real"].copy()

    def console():
        while state["run"]:
            try:
                cmd = input("> ")
            except EOFError:
                break
            state["run"] = handle_command(cmd, units, get_real, args.units, sim)

    threading.Thread(target=console, daemon=True).start()
    try:
        with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
            while state["run"] and v.is_running():
                pos = robot.bus.sync_read("Present_Position")
                state["real"] = np.array([pos[j] for j in S.JOINTS], float)
                sim.d.qpos[sim.qadr] = units.to_sim(state["real"])
                mujoco.mj_forward(sim.m, sim.d)
                v.sync()
                time.sleep(0.02)
    finally:
        state["run"] = False
        robot.disconnect()
        print("\n終了しました。")


# ---------------- replay ----------------
def to_uint8_hwc(img):
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def replay_frames(sim, units, states, real_tops, order, size=(320, 240)):
    """実機の関節値 states(データセットの列順)をシミュレーションで再現し、比較画像を返すジェネレータ。"""
    import cv2
    w, h = size
    side = mujoco.Renderer(sim.m, h, w)
    cam, opt = side_camera(), side_option()
    for st, real_top in zip(states, real_tops):
        real = np.empty(6)
        real[order] = st
        sim.d.qpos[sim.qadr] = units.to_sim(real)
        mujoco.mj_forward(sim.m, sim.d)
        sim_top = sim.render()["top"]
        side.update_scene(sim.d, cam, scene_option=opt)
        a = cv2.resize(real_top, (w, h))
        b = cv2.resize(sim_top, (w, h))
        blend = (a.astype(np.float32) * 0.5 + b.astype(np.float32) * 0.5).astype(np.uint8)
        yield np.concatenate([np.concatenate([a, b], 1), np.concatenate([blend, side.render()], 1)], 0)
    side.close()


def report(sim, units, states, order):
    """最初の姿勢と、可動範囲を超えた関節を表示する。"""
    real = np.empty((len(states), 6))
    real[:, order] = states
    q = units.to_sim(real)
    lo, hi = sim.m.jnt_range[[sim.m.joint(j).id for j in S.JOINTS]].T
    print("\n最初の姿勢 [rad](gen_sim_dataset.py の MOTION['home'] に使う):")
    print("  home=[" + ", ".join(f"{v:.2f}" for v in q[0]) + "]")
    print("\n関節ごとの範囲(実機の値 → シミュレーションの角度[度]、シミュレーションの可動範囲[度]):")
    bad = []
    for i, j in enumerate(S.JOINTS):
        qmin, qmax = math.degrees(q[:, i].min()), math.degrees(q[:, i].max())
        out = q[:, i].min() < lo[i] - 0.05 or q[:, i].max() > hi[i] + 0.05
        if out:
            bad.append(j)
        print(f"  {j:14s} 実機 {real[:, i].min():+7.1f}〜{real[:, i].max():+7.1f}  →  {qmin:+6.0f}〜{qmax:+6.0f}度"
              f"  (可動範囲 {math.degrees(lo[i]):+.0f}〜{math.degrees(hi[i]):+.0f}){'  ※範囲外' if out else ''}")
    if bad:
        print("\n※範囲外の関節があります。数度なら、実機の可動範囲がモデルより少し広いだけなので問題ありません"
              "(so101_sim.py の joint_range_override で広げられます)。10度以上なら向き(flip)かずれ(zero)を疑ってください。")


def press_point(sim, units, states, order):
    """指先が土台から水平方向に一番遠くまで伸びたフレームを「押した位置」とみなす(壁のボタンを押すタスク用)。"""
    ik = S.IK(sim)
    best = None
    for k, st in enumerate(states):
        real = np.empty(6)
        real[order] = st
        p, a = ik.fk(units.to_sim(real))
        reach = math.hypot(p[0], p[1])
        if best is None or reach > best[0]:
            pitch = math.degrees(math.asin(np.clip(a[2], -1, 1)))      # 指先の上下の向き(+ が上向き)
            yaw = math.degrees(math.atan2(a[1], a[0]))                # 指先の左右の向き(+ が左)
            best = (reach, k, p, pitch, yaw)
    return best[1:]


def replay(args):
    import imageio.v2 as imageio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.root or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "lerobot", args.repo_id)
    eps = [int(e) for e in str(args.episode).split(",")]
    ds = LeRobotDataset(args.repo_id, root=root, episodes=eps)
    names = ds.meta.features["observation.state"]["names"]
    order = [S.JOINTS.index(n.removesuffix(".pos")) for n in names]
    top_key = "observation.images.top"
    info_shape = ds.meta.features[top_key]["shape"]
    sim = S.Sim(args.model_dir, dict(img_h=info_shape[0], img_w=info_shape[1]))
    units = S.Units(args.units)
    print(f"単位変換: {units.source}")

    per_ep = {e: ([], []) for e in eps}
    for i in range(0, len(ds), args.stride):
        item = ds[i]
        e = int(item["episode_index"])
        if e not in per_ep:
            continue
        per_ep[e][0].append(item["observation.state"].numpy())
        if e == eps[0] and not args.no_video:
            per_ep[e][1].append(to_uint8_hwc(item[top_key]))

    print("\n押した位置の推定(土台が原点、+x が正面、+y が左、+z が上、単位 cm。指先が一番前に伸びたところ):")
    pts = []
    for e in eps:
        r = press_point(sim, units, np.array(per_ep[e][0]), order)
        if r is None:
            print(f"  エピソード {e}: 見つかりませんでした")
            continue
        k, p, pitch, yaw = r
        pts.append(p)
        print(f"  エピソード {e}: x={p[0] * 100:5.1f}  y={p[1] * 100:+5.1f}  z={p[2] * 100:5.1f}"
              f"   指先の向き 上下{pitch:+4.0f}度 左右{yaw:+4.0f}度   時刻 {k * args.stride / ds.fps:4.1f}s")
    if len(pts) > 1:
        p = np.mean(pts, axis=0)
        print(f"  平均: x={p[0] * 100:5.1f}  y={p[1] * 100:+5.1f}  z={p[2] * 100:5.1f}"
              f"   ばらつき(標準偏差) {np.std(pts, axis=0)[:2].max() * 100:.1f}cm")
    if pts:
        import env_builder as E
        pc = E.panel_pos_from_button(np.mean(pts, axis=0))
        print(f"  → env_builder.py の panel_pos(パネル中心)に入れる値: [{pc[0]:.3f}, {pc[1]:.3f}, {pc[2]:.3f}]")

    states = np.array(per_ep[eps[0]][0])
    print(f"\n--- エピソード {eps[0]} ---")
    report(sim, units, states, order)
    if args.no_video:
        return
    out = args.out or f"replay_ep{eps[0]}.mp4"
    with imageio.get_writer(out, fps=ds.fps / args.stride, macro_block_size=8) as w:
        for frame in replay_frames(sim, units, states, per_ep[eps[0]][1], order):
            w.append_data(frame)
    print(f"\n動画: {os.path.abspath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "replay"])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--units", default="units.json")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--repo_id", default="local/ac_power_button")
    ap.add_argument("--root", default="")
    ap.add_argument("--episode", default="0", help="エピソード番号。カンマ区切りで複数(動画は最初の1つだけ)")
    ap.add_argument("--no_video", action="store_true", help="動画を作らない(位置の推定だけ)")
    ap.add_argument("--stride", type=int, default=2, help="何フレームごとに描くか")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    live(args) if args.mode == "live" else replay(args)


if __name__ == "__main__":
    main()