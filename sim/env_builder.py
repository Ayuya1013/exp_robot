"""SO-101 とエアコンのコントロールパネル(control_panel.xml)を1つの MuJoCo シーンにまとめる。

座標系はロボットの土台が原点、+x が正面、+y が左、+z が上(単位は m)。
control_panel.xml はパネル正面が -y、押し込みが +y の向きで作ってあるので、
ここでは「パネルの正面がロボットを向く」ように z 軸まわりに -90 度回してから置く。
ハイパーパラメータ:
    panel_pos     : パネル中心(control_panel_root)の位置 [x, y, z]
    panel_rpy_deg : 「ロボットを向いた向き」からの回転 [roll, pitch, yaw](度)。0, 0, 0 で壁がロボットの正面に平行
必要なもの: mujoco 3.3 以降(MjSpec.attach を使う)、SO-ARM100 リポジトリの Simulation/SO101。

使い方:
    python env_builder.py --model_dir ~/SO-ARM100/Simulation/SO101                       # ビューアで表示
    python env_builder.py --model_dir ... --panel_pos 0.35 -0.03 0.20 --panel_rpy 0 0 10  # 位置と向きを変える
    python env_builder.py --model_dir ... --save_cams                                    # top と wrist の画像を保存
    python env_builder.py --limits_from_stats <データセット>/meta/stats.json               # 実データから関節の範囲を作る
"""
import argparse
import json
import math
import os
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# ===== ロボットのモデル =====
# so101_new_calib_camera.xml は so101_new_calib.xml に手先カメラの取り付け部品とカメラ(32x32mm の UVC 基板)を足したもの。
# 実機と同じく、キャリブレーションの姿勢で爪の左側(+y)、手首を -90 度回すと上側にカメラが来る。なければ素のモデルを使う
ROBOT_XML = ["so101_new_calib_camera.xml", "so101_new_calib.xml"]

# ===== パネルと壁 =====
PANEL = dict(
    panel_xml=os.path.join(HERE, "control_panel.xml"),
    # パネルの置き方は2通り。button_pos を基準にするのがおすすめ(パネルの xml のボタン配置を直しても、押すボタンの位置は変わらない)
    #   button_pos: 押すボタン(target_button)の前面中心の位置。panel_pos が None のとき、ここから panel_pos を計算する
    #   panel_pos : パネル中心の位置を直接決める(指定すると button_pos より優先)
    # 2026/10/25 live で爪の先を「運転/停止」に当てて測った値: 肩の回転軸から 前 28.4、左 -8.5 cm、高さ 24.0 cm
    #   (机に触れた確認で、関節の対応のずれは高さ1cm以内、前後0.4cm以内)。回転軸はモデルの原点から x=+0.0388
    button_pos=[0.0388 + 0.284, -0.085, 0.240],
    panel_pos=None,
    # (参考)定規での実測(2026/10/25): 台の前端から壁まで 277mm、左右 -50mm、台の底面からパネル中心まで 270mm、パネルの厚み 18mm。
    #   台の前端はモデルの x=0.0646。button_pos から計算すると、壁は台の前端から 280mm、パネル中心の高さは 258mm で、
    #   奥行きは 3mm、高さは 12mm の差(机に触れた確認で、順運動学の高さの誤差は 1cm 程度)
    panel_rpy_deg=[0.0, 0.0, 0.0],
    target_button="power",             # 押すボタン(btn_<名前>)。押す位置の目印 press_target をここに付ける
    button_stiffness=None,             # ばね [N/m]。None なら control_panel.xml の値(500)を使う
    wall_size=[0.5, 0.01, 0.5],        # 壁の半分の大きさ(パネル座標で 横, 厚み, 縦)
    # 壁の表面が、パネルの正面から何 m 奥にあるか。None ならパネル本体の裏面に接する(= 厚み 18mm)。
    # 2026/10/25 live の wall で測ると 6mm だった(パネルの残りは壁に埋まっている)。台の前端から壁まで定規で 257mm とも合う
    wall_depth=0.006,
    # 右の壁: パネルの右端から gap だけ右に、パネルの壁と直角に立ち、ロボットの方へ length だけ出ている(俯瞰カメラはその先端に付いている)
    # 2026/10/26 実測: パネルの右端から壁まで約 1.2cm。length は俯瞰カメラの較正の位置から(パネルの壁からカメラまで 25.3cm)
    side_wall=dict(gap=0.012, length=0.253, thickness=0.02, height=1.2, rgba=[0.93, 0.93, 0.92, 1]),
)
# ===== カメラ(cameras.json があれば上書き。tune_camera.py で作る) =====
CAMERAS = dict(
    img_w=640, img_h=480,
    # 俯瞰(C920): ロボットの台の右上にあり、壁に正対している(視線が +x と平行)。位置は仮の値で、較正で決める
    top=dict(pos=[-0.05, -0.10, 0.20], lookat=[0.33, -0.10, 0.20], up=[0, 0, 1], fovy=45.0, rot180=False),
    # 手先: gripper ボディの座標系(指先の向きが -z、カメラのある側が +y)。
    #   位置は so101_new_calib_camera.xml のカメラ基板の前面の中心。向きもそのモデルの基板の向き(爪の方へ約25度内向き)。
    #   up は「手首を -90 度回してカメラが上に来たとき、画像の上が天井」になる向き。画角は仮の値で、較正で決める
    wrist=dict(pos=[0.003, 0.0729, -0.0186], lookat=[0.003, 0.0305, -0.1092], up=[0, 1, 0], fovy=65.0, rot180=False),
)
# ===== 関節の範囲(limits.json があれば上書き) =====
LIMITS = dict(
    # 実機の値(LeRobot の度、グリッパーは 0〜100)で書く。書いていない関節はモデルの範囲のまま
    joint_limits_real={},
    # モデルの範囲そのものを広げる [rad](実機の待機姿勢で shoulder_lift が約 -105 度まで行くため)
    model_range_override={"shoulder_lift": [-1.92, 1.74533]},
    # 指先(gripperframe)が出てよい範囲 [m]。None なら確かめない
    tcp_box=dict(min=[-0.10, -0.40, 0.0], max=[0.60, 0.40, 0.55]),
)
# =====================


def rpy_to_quat(rpy_deg):
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.radians(rpy_deg), "xyz")
    return q


def panel_quat(rpy_deg):
    """パネルの向き。z 軸まわり -90 度(正面をロボットに向ける)に、ハイパーパラメータの回転を重ねる。"""
    base = np.array([math.cos(-math.pi / 4), 0, 0, math.sin(-math.pi / 4)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, rpy_to_quat(rpy_deg), base)
    return q


def _xyaxes(pos, lookat, up):
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(f, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return np.concatenate([x, np.cross(x, f)])


def cam_quat(pos, lookat, up):
    xy = _xyaxes(pos, lookat, up)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([xy[:3], xy[3:], np.cross(xy[:3], xy[3:])]).flatten())
    return q


def _set_first(obj, attr, val):
    """mujoco の版によって配列のこともスカラーのこともある属性に、先頭の値を入れる。"""
    cur = getattr(obj, attr)
    if np.ndim(cur):
        arr = np.array(cur, float)
        arr[0] = val
        setattr(obj, attr, arr)
    else:
        setattr(obj, attr, val)


def load_json(path, default):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {**default, **json.load(f)}
    return dict(default)


def target_local(panel_spec, button):
    """パネル座標で、押すボタンの前面中心の位置(押していないとき)。"""
    b = panel_spec.body(f"btn_{button}")
    site = panel_spec.site(f"site_{button}")
    return np.array(b.pos) + np.array(site.pos)


def panel_pos_from_button(button_xyz, rpy_deg=(0, 0, 0), button="power", panel_xml=PANEL["panel_xml"]):
    """ボタン前面の位置(例: check_alignment.py の推定値)から、パネル中心の位置を求める。"""
    ps = mujoco.MjSpec.from_file(panel_xml)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, panel_quat(rpy_deg))
    return np.asarray(button_xyz, float) - R.reshape(3, 3) @ target_local(ps, button)


def joint_limits_sim(limits_real, units):
    """実機の値の範囲を、シミュの角度 [rad] の範囲に直す(units は so101_sim.Units か None)。"""
    out = {}
    for j, (lo, hi) in limits_real.items():
        i = JOINTS.index(j)
        if units is None:
            a, b = (math.radians(lo), math.radians(hi)) if j != "gripper" else (lo / 50.3 - 0.1745, hi / 50.3 - 0.1745)
        else:
            a = (lo - units.offset[i]) / units.scale[i]
            b = (hi - units.offset[i]) / units.scale[i]
        out[j] = [min(a, b), max(a, b)]  # 向きが反対(scale<0)なら入れ替える
    return out


def lens_model(cam, img_w, img_h, scale=1.5, margin=1.04):
    """実カメラの内部パラメータ(cam["intrinsics"] の json)があれば、ゆがみを再現するための情報を作る。
    シミュではゆがみのない(ピンホール)画像を少し広い画角で描き、cv2.remap で実カメラと同じゆがみ・主点にする。
    返り値: dict(fovy, W, H, mapx, mapy) か None"""
    path = cam.get("intrinsics")
    if not path:
        return None
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(path):
        print(f"注意: {path} がないので、ゆがみなしで描きます")
        return None
    import cv2
    with open(path, encoding="utf-8") as f:
        intr = json.load(f)
    K = np.array(intr["pinhole"]["K"], float)
    dist = np.array(intr["pinhole"]["dist"], float)
    sw, sh = intr["size"]
    if (sw, sh) != (img_w, img_h):  # 画像の大きさが違えば、K を合わせる
        K[0] *= img_w / sw
        K[1] *= img_h / sh
    uu, vv = np.meshgrid(np.arange(img_w, dtype=np.float32), np.arange(img_h, dtype=np.float32))
    pts = np.stack([uu, vv], -1).reshape(-1, 1, 2)
    n = cv2.undistortPoints(pts, K, dist).reshape(img_h, img_w, 2)  # 正規化座標(x 右、y 下)
    ax, ay = np.abs(n[..., 0]).max() * margin, np.abs(n[..., 1]).max() * margin
    H = int(round(img_h * scale))
    f = H / 2 / ay
    W = int(round(2 * f * ax))
    mapx = (f * n[..., 0] + W / 2).astype(np.float32)
    mapy = (f * n[..., 1] + H / 2).astype(np.float32)
    return dict(fovy=math.degrees(2 * math.atan(ay)), W=W, H=H, mapx=mapx, mapy=mapy)


def build_spec(model_dir, panel=None, cameras=None, limits=None, units=None):
    """シーンの MjSpec を作る(compile 前)。返り値: (spec, 使った設定)"""
    p = {**PANEL, **load_json(os.path.join(HERE, "panel.json"), {}), **(panel or {})}  # panel.json は live の btn / move で作る
    if p.get("panel_pos") is None:
        p["panel_pos"] = panel_pos_from_button(p["button_pos"], p["panel_rpy_deg"], p["target_button"],
                                               p["panel_xml"]).tolist()
    import copy
    c = copy.deepcopy(cameras or load_json(os.path.join(HERE, "cameras.json"), CAMERAS))  # 元の設定は書き換えない
    lim = limits or load_json(os.path.join(HERE, "limits.json"), LIMITS)

    robot_xml = next((os.path.join(model_dir, f) for f in ROBOT_XML if os.path.exists(os.path.join(model_dir, f))), None)
    if robot_xml is None:
        raise FileNotFoundError(f"{model_dir} に {ROBOT_XML} のどれもありません")
    spec = mujoco.MjSpec.from_file(robot_xml)
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = max(c["img_w"], 640)
    spec.visual.global_.offheight = max(c["img_h"], 480)
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]

    # 関節の範囲(モデルの範囲を広げてから、実機の値で決めた範囲を入れる)
    ranges = {j: list(r) for j, r in lim["model_range_override"].items()}
    ranges.update(joint_limits_sim(lim["joint_limits_real"], units))
    for j, (lo, hi) in ranges.items():
        spec.joint(j).range = [lo, hi]
        spec.actuator(j).ctrlrange = [lo, hi]  # 位置の目標も範囲内に丸められる

    wb = spec.worldbody
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], rgba=[0.55, 0.5, 0.45, 1])
    wb.add_light(name="key", pos=[0.0, -0.6, 1.2], dir=[0.3, 0.5, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True)

    # パネルと壁を「panel_mount」ボディにまとめる(実行中は model.body_pos / body_quat で動かせる)
    mount = wb.add_body(name="panel_mount", pos=list(p["panel_pos"]), quat=list(panel_quat(p["panel_rpy_deg"])))
    ps = mujoco.MjSpec.from_file(p["panel_xml"])
    root = ps.body("control_panel_root")
    # 壁(パネル座標: 正面が -y)。パネル本体と重なってもよい(固定物どうしは当たり判定をしない)
    board = ps.geom("panel_board")
    if p.get("wall_depth") is None:
        surface = board.pos[1] + board.size[1]               # パネル本体の裏面
    else:
        surface = board.pos[1] - board.size[1] + p["wall_depth"]  # パネルの正面から wall_depth 奥
    w = p["wall_size"]
    mount.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, size=w, pos=[0, surface + w[1], 0],
                   rgba=[0.88, 0.87, 0.83, 1], group=1, contype=2, conaffinity=1)
    sw = p.get("side_wall")
    if sw:  # パネル座標: x が右、y が奥(押す向き)、z が上
        half_w = board.size[0]
        mount.add_geom(name="side_wall", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[sw["thickness"] / 2, sw["length"] / 2, sw["height"] / 2],
                       pos=[half_w + sw["gap"] + sw["thickness"] / 2, surface - sw["length"] / 2, 0],
                       rgba=list(sw["rgba"]), group=1, contype=2, conaffinity=1)
    if np.linalg.norm(root.pos) > 1e-9:
        print(f"注意: control_panel.xml の control_panel_root の pos が {list(root.pos)} です。"
              "panel_pos からさらにずれるので、0 0 0 にしてください。")
    root.add_site(name="press_target", pos=list(target_local(ps, p["target_button"])), size=[0.002, 0, 0], group=4)
    # 当たり判定・ばね・押し込みの上限は control_panel.xml の値をそのまま使う。
    # button_stiffness を指定したときだけ、ばねを上書きする(ランダム化などで使う)
    if p["button_stiffness"] is not None:
        for b in root.bodies:
            for j in b.joints:
                _set_first(j, "stiffness", p["button_stiffness"])
    spec.attach(ps, prefix="panel/", frame=mount.add_frame())
    # 目印の site(ボタンの押す位置、指先の点など)は画像に写らないようにする(グループ 4 は描かない)。
    # 写っていると、ボタンの前面が灰色の site で覆われて暗く見えていた
    for st in spec.sites:
        st.group = 4

    # カメラ(内部パラメータがあれば、ゆがみ用に少し広い画角・大きい画像で描く)
    for name in ("top", "wrist"):
        c[name]["_lens"] = lens_model(c[name], c["img_w"], c["img_h"])
        if c[name]["_lens"] is not None:
            spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, c[name]["_lens"]["W"])
            spec.visual.global_.offheight = max(spec.visual.global_.offheight, c[name]["_lens"]["H"])
    fov = lambda name: c[name]["_lens"]["fovy"] if c[name]["_lens"] else c[name]["fovy"]
    t = c["top"]
    wb.add_camera(name="top", pos=list(t["pos"]), quat=list(cam_quat(t["pos"], t["lookat"], t["up"])), fovy=fov("top"))
    wr = c["wrist"]
    spec.body("gripper").add_camera(name="wrist", pos=list(wr["pos"]),
                                    quat=list(cam_quat(wr["pos"], wr["lookat"], wr["up"])), fovy=fov("wrist"))
    return spec, dict(panel=p, cameras=c, limits=lim)


def load_environment(model_dir, panel_pos=None, panel_rpy_deg=None, **kw):
    """ハイパーパラメータからモデルを作る。返り値: (model, 設定)"""
    panel = dict(kw.pop("panel", {}) or {})
    if panel_pos is not None:
        panel["panel_pos"] = panel_pos
    if panel_rpy_deg is not None:
        panel["panel_rpy_deg"] = panel_rpy_deg
    spec, cfg = build_spec(model_dir, panel=panel, **kw)
    return spec.compile(), cfg


def set_panel_pose(model, panel_pos, panel_rpy_deg=(0, 0, 0)):
    """作り直さずにパネル(と壁)を動かす。このあと mj_forward か mj_step を呼ぶ。"""
    bid = model.body("panel_mount").id
    model.body_pos[bid] = panel_pos
    model.body_quat[bid] = panel_quat(panel_rpy_deg)


def tcp_in_box(model, data, box):
    if box is None:
        return True
    p = data.site("gripperframe").xpos
    return bool(np.all(p >= box["min"]) and np.all(p <= box["max"]))


_RENDERERS = {}


def render_cams(model, data, cameras, renderer=None):
    """top と wrist を描く。内部パラメータがあるカメラは、実カメラと同じゆがみと主点にしてから返す。"""
    out = {}
    for name in ("top", "wrist"):
        lens = cameras[name].get("_lens")
        if lens is None:
            r = renderer or _RENDERERS.setdefault((id(model), "plain"), mujoco.Renderer(model, cameras["img_h"], cameras["img_w"]))
            r.update_scene(data, camera=name)
            img = r.render().copy()
        else:
            import cv2
            key = (id(model), lens["W"], lens["H"])
            r = _RENDERERS.get(key)
            if r is None:
                r = _RENDERERS[key] = mujoco.Renderer(model, lens["H"], lens["W"])
            r.update_scene(data, camera=name)
            img = cv2.remap(r.render(), lens["mapx"], lens["mapy"], cv2.INTER_AREA if False else cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        if cameras[name].get("rot180"):
            img = np.ascontiguousarray(img[::-1, ::-1])
        out[name] = img
    return out


def limits_from_stats(stats_path, margin, out_path):
    """実データの meta/stats.json の action の最小・最大に余裕を足して limits.json に書く。"""
    with open(stats_path, encoding="utf-8") as f:
        st = json.load(f)["action"]
    lim = {}
    for i, j in enumerate(JOINTS):
        lo, hi = float(st["min"][i]) - margin, float(st["max"][i]) + margin
        if j == "gripper":
            lo, hi = max(lo, 0.0), min(hi, 100.0)
        lim[j] = [round(lo, 1), round(hi, 1)]
    data = load_json(out_path, {})
    data["joint_limits_real"] = lim
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    for j, (lo, hi) in lim.items():
        print(f"  {j:14s} {lo:+7.1f} 〜 {hi:+7.1f}")
    print(f"保存しました: {os.path.abspath(out_path)}(関節の順は {JOINTS} を前提にしている)")


def snapshot(model_dir, home, out):
    """今の設定(panel.json、cameras.json、units.json)のシーンを、3方向 x 2姿勢で1枚の画像にする。"""
    import imageio.v2 as imageio
    import so101_sim as S
    import cv2
    sim = S.Sim(model_dir)
    ik = S.IK(sim)
    home = np.array(home)
    sim.reset(home)
    bt = sim.button_top()
    axis = sim.d.xanchor[sim.m.joint("shoulder_pan").id]
    r = np.array([bt[0] - axis[0], bt[1] - axis[1], 0.0])
    r /= np.linalg.norm(r)
    q_press, _, _ = ik.solve(np.r_[home[:4], -1.5, home[5]], bt + r * 0.001, r)
    big = mujoco.Renderer(sim.m, 480, 640)
    rows = []
    for q in (home, q_press):
        sim.reset(q)
        cams = sim.render()
        views = []
        for az, el, look, hide in ((-35, -15, [0.22, -0.05, 0.18], False), (90, -5, [0.18, -0.05, 0.18], True)):
            c = mujoco.MjvCamera()
            c.lookat[:], c.distance, c.azimuth, c.elevation = look, 0.75, az, el
            opt = mujoco.MjvOption()
            if hide:  # 横からの図では壁(グループ1)を隠す
                opt.geomgroup[1] = 0
            big.update_scene(sim.d, c, scene_option=opt)
            views.append(big.render())
        tiles = [cv2.resize(v, (640, 480)) for v in (views[0], views[1], cams["top"], cams["wrist"])]
        rows.append(np.concatenate(tiles, axis=1))
    img = np.concatenate(rows, axis=0)
    for i, label in enumerate(["overview", "side (walls hidden)", "top camera", "wrist camera"]):
        cv2.putText(img, label, (i * 640 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    imageio.imwrite(out, img[::2, ::2])
    print(f"保存しました: {os.path.abspath(out)}(上の段: 待機姿勢、下の段: 押す瞬間)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    ap.add_argument("--panel_pos", type=float, nargs=3, help="パネル中心の位置(button_pos より優先)")
    ap.add_argument("--button_pos", type=float, nargs=3, help="押すボタンの前面中心の位置")
    ap.add_argument("--panel_rpy", type=float, nargs=3)
    ap.add_argument("--home", type=float, nargs=6, default=[0.0, -1.83, 1.58, 0.97, 0.0, 0.02], help="腕の姿勢 [rad]")
    ap.add_argument("--save_cams", action="store_true", help="top と wrist の画像を保存して終わる")
    ap.add_argument("--snapshot", default="", help="全体・俯瞰・手先の画像を、待機姿勢と押す瞬間の2段で1枚にまとめて保存する(例: scene.png)")
    ap.add_argument("--limits_from_stats", default="", help="実データの meta/stats.json")
    ap.add_argument("--margin", type=float, default=5.0, help="limits_from_stats の余裕 [度]")
    args = ap.parse_args()

    if args.limits_from_stats:
        limits_from_stats(args.limits_from_stats, args.margin, os.path.join(HERE, "limits.json"))
        return

    try:
        import so101_sim
        units = so101_sim.Units(os.path.join(HERE, "units.json"))
    except ImportError:
        units = None
    panel = {"button_pos": args.button_pos} if args.button_pos else None
    model, cfg = load_environment(args.model_dir, args.panel_pos, args.panel_rpy, units=units, panel=panel)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(j).id] for j in JOINTS]
    data.qpos[qadr] = args.home
    data.ctrl[[model.actuator(j).id for j in JOINTS]] = args.home
    mujoco.mj_forward(model, data)
    print(f"パネル中心 {np.round(data.body('panel_mount').xpos, 3)}  押す位置 {np.round(data.site('panel/press_target').xpos, 3)}")

    if args.snapshot:
        snapshot(args.model_dir, args.home, args.snapshot)
        return

    if args.save_cams:
        import imageio.v2 as imageio
        imgs = render_cams(model, data, cfg["cameras"])
        for k, v in imgs.items():
            imageio.imwrite(f"cam_{k}.png", v)
        print("保存しました: cam_top.png, cam_wrist.png")
        return

    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))


if __name__ == "__main__":
    main()