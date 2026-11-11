"""SO-101 の MuJoCo シミュレーション部品(シーン・ランダム化・IK・単位変換)。

シーン(ロボット、壁、コントロールパネル、カメラ、関節の範囲)は env_builder.py で作る。
ここでは、それを動かすためのクラス(Sim)、IK、実機の値との単位変換(Units)を持つ。
座標系: ロボットの土台が原点、+x が正面、+z が上。単位は m とラジアン。
"""

import json
import math
import os

import env_builder as E
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOINTS = E.JOINTS
IK_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]  # IK で動かす関節

# ===== ランダム化(ドメインランダム化)=====
# 今の設定を中心にして、エピソードごとにばらつかせる。強さは θ(0 でばらつきなし、2 で2倍)。
#  ・位置・向き(測った値の誤差): 正規分布。値は θ=1 の標準偏差。±2.5σ で切る
#  ・見た目(色・明るさ・照明): 中心が正しいとは限らないので、範囲を偏りなく覆う一様分布
#      倍率で効くもの(明るさ、照明の強さ)は対数一様分布。値 r は θ=1 で 1/r〜r 倍(θ 倍すると範囲が広がる)
#      足し算で効くもの(色味、照明の位置)は一様分布。値 w は θ=1 で ±w
# random.json があれば上書きする。色の中心は colors.json(camcal.py colors で実画像から作る)
RANDOM = dict(
    # 正規分布(標準偏差)
    panel_pos_m=[0.005, 0.008, 0.008],  # パネル(と壁)の位置 x, y, z。ロボットの置き直しのずれ程度
    panel_rpy_deg=[0.0, 0.7, 1.0],  # パネルの向き。実機の壁は台と平行なので小さく
    top_cam_pos_m=0.004,
    top_cam_rot_deg=0.5,
    top_cam_fovy_deg=0.4,
    wrist_cam_pos_m=0.002,
    wrist_cam_rot_deg=1.0,
    wrist_cam_fovy_deg=0.5,  # 手先は較正の残りのずれ(約1度)程度
    # 対数一様分布(倍率 r: θ=1 で 1/r〜r 倍)
    brightness_ratio=1.25,  # 部品ごとの色の明るさ(0.8〜1.25 倍)
    light_ratio=1.6,  # 照明の強さ(0.63〜1.6 倍)
    headlight_ratio=1.33,  # 全体の明るさ(0.75〜1.33 倍)
    # 一様分布(幅 ±w)
    hue=0.06,  # 部品ごとの色味(R, G, B を別々に、倍率の ±)
    light_pos_m=0.20,  # 照明の位置
)
CFG = dict(img_w=640, img_h=480)  # 実データの info.json があれば、その大きさに合わせる
# =====================================


ROBOT_PART_RGB = [
    0.90,
    0.90,
    0.88,
]  # 実機の3Dプリント部品は白(モデルの黄色のままにしない)。colors.json の robot_parts で上書き


def apply_colors(m, path):
    """基準の色を決める。3Dプリント部品(モデルで黄色の材質)は白にし、colors.json(geom の名前 → RGB)があれば上書きする。
    返り値: 使ったファイルか None
    """
    colors = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            colors = json.load(f)
    part = colors.get("robot_parts", ROBOT_PART_RGB)
    for i in range(
        m.nmat
    ):  # モデルの黄色(1, 0.82, 0.12)の材質 = 3Dプリント部品。モーターやカメラ基板(黒)はそのまま
        if np.allclose(m.mat_rgba[i, :3], [1.0, 0.82, 0.12], atol=0.02):
            m.mat_rgba[i, :3] = part
    for name, rgb in colors.items():
        if name.startswith("_") or name == "robot_parts":
            continue
        try:
            m.geom_rgba[m.geom(name).id, :3] = rgb
        except KeyError:
            print(f"注意: colors.json の {name} はシーンにありません")
    return path if colors else None


def robot_part_materials(m):
    """3Dプリント部品の材質の番号(色を合わせるときに使う)。"""
    return [
        i
        for i in range(m.nmat)
        if m.material(i).name.endswith("_material")
        and not m.material(i).name.startswith(("sts3215", "wrist_camera_so101"))
    ]


def load_random(path=os.path.join(HERE, "random.json")):
    r = json.loads(json.dumps(RANDOM))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            r.update(json.load(f))
    return r


def find_model_xml(model_dir):
    p = os.path.join(model_dir, "so101_new_calib.xml")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} がありません。SO-ARM100 を git clone して、Simulation/SO101 のフォルダを指定してください。"
        )
    return p


class Sim:
    """env_builder.py のシーンを動かす。パネル位置のランダム化、押し込みの判定、2カメラの描画。"""

    def __init__(
        self, model_dir, cfg=None, render=True, units_path=os.path.join(HERE, "units.json"), theta=1.0
    ):
        self.cfg = dict(CFG, **(cfg or {}))
        self.theta = theta
        self.rand = load_random()
        c = self.cfg
        find_model_xml(model_dir)
        cams = E.load_json(os.path.join(E.HERE, "cameras.json"), E.CAMERAS)
        cams["img_w"], cams["img_h"] = c["img_w"], c["img_h"]
        units = Units(units_path) if os.path.exists(units_path) else None
        spec, self.env = E.build_spec(model_dir, cameras=cams, units=units)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        btn = self.env["panel"]["target_button"]
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        bj = m.joint(f"panel/joint_{btn}")
        self.btn_qadr = m.jnt_qposadr[bj.id]
        c["button_travel"] = float(m.jnt_range[bj.id][1])
        self.mount = m.body("panel_mount").id
        self.lcd_geom = m.geom("panel/lcd_screen").id
        self.target_geom = m.geom(f"panel/geom_{btn}").id
        panel_bodies = {i for i in range(m.nbody) if m.body(i).name.startswith("panel")}
        # 押すボタン以外のパネルと壁(ほかのボタンも含む)
        self.dev_geoms = {
            i for i in range(m.ngeom) if m.geom_bodyid[i] in panel_bodies and i != self.target_geom
        }
        self.robot_geoms = {
            i for i in range(m.ngeom) if m.geom_bodyid[i] not in panel_bodies and m.geom_bodyid[i] != 0
        }
        self.tcp_box = self.env["limits"].get("tcp_box")
        self.colors_source = apply_colors(m, os.path.join(HERE, "colors.json"))
        self._base = dict(
            cam_pos=m.cam_pos.copy(),
            cam_quat=m.cam_quat.copy(),
            cam_fovy=m.cam_fovy.copy(),
            geom_rgba=m.geom_rgba.copy(),
            mat_rgba=m.mat_rgba.copy(),
            light_diffuse=m.light_diffuse.copy(),
            light_pos=m.light_pos.copy(),
            headlight=m.vis.headlight.diffuse.copy(),
        )
        self._lcd_off = m.geom_rgba[self.lcd_geom].copy()
        mujoco.mj_forward(m, self.d)
        self.renderer = mujoco.Renderer(m, c["img_h"], c["img_w"]) if render else None

    # ---------- 状態 ----------
    def q(self):
        return self.d.qpos[self.qadr].copy()

    def button_depth(self):
        return float(self.d.qpos[self.btn_qadr])

    def button_top(self):
        """押していないときの、押すボタンの前面中心。"""
        return self.d.site("panel/press_target").xpos.copy()

    def push_dir(self):
        """ボタンを押し込む向き(パネル座標の +y、壁に向かう単位ベクトル)。"""
        return self.d.xmat[self.mount].reshape(3, 3)[:, 1].copy()

    def set_lcd(self, on):
        """押したあと画面が光る(実機ではバックライトが点く)。"""
        self.m.geom_rgba[self.lcd_geom] = [0.55, 0.8, 1.0, 1] if on else self._lcd_off

    def reset(self, q0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = q0
        self.d.ctrl[self.act] = q0
        self.set_lcd(False)
        mujoco.mj_forward(self.m, self.d)

    def step(self, target, dt, near=0.03):
        """関節目標 target を与えて dt 秒進める。
        押すボタンから near [m] より離れた所で、ロボットが壁・パネル・ほかのボタンに当たるか、
        指先が tcp_box の外に出たら True を返す。
        """
        self.d.ctrl[self.act] = target
        hit = False
        top = self.button_top()
        for _ in range(max(1, round(dt / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)
            for k in range(self.d.ncon):
                c = self.d.contact[k]
                g1, g2 = c.geom1, c.geom2
                if (g1 in self.dev_geoms and g2 in self.robot_geoms) or (
                    g2 in self.dev_geoms and g1 in self.robot_geoms
                ):
                    if np.linalg.norm(c.pos - top) > near:
                        hit = True
        if not E.tcp_in_box(self.m, self.d, self.tcp_box):
            hit = True
        return hit

    def render(self):
        return E.render_cams(self.m, self.d, self.env["cameras"], self.renderer)

    # ---------- ランダム化 ----------
    def randomize(self, rng, visual=True):
        """今の設定を中心に、θ 倍の標準偏差でばらつかせる。返り値: 選んだ値(記録用の dict)"""
        R, m, b, th = self.rand, self.m, self._base, self.theta

        def g(sigma, size=None):  # 正規分布: 平均 0、標準偏差 θσ(±2.5σ で切る)
            return np.clip(rng.normal(0.0, 1.0, size), -2.5, 2.5) * np.asarray(sigma, float) * th

        def u(w, size=None):  # 一様分布: ±θw
            return rng.uniform(-1.0, 1.0, size) * np.asarray(w, float) * th

        def logu(r, size=None):  # 対数一様分布: θ=1 で 1/r〜r 倍
            return np.exp(rng.uniform(-1.0, 1.0, size) * math.log(r) * th)

        p = self.env["panel"]
        d_pos, d_rpy = g(R["panel_pos_m"], 3), g(R["panel_rpy_deg"], 3)
        E.set_panel_pose(m, np.array(p["panel_pos"]) + d_pos, np.array(p["panel_rpy_deg"]) + d_rpy)
        params = dict(
            theta=th, panel_dpos_mm=(d_pos * 1000).round(2).tolist(), panel_drpy_deg=d_rpy.round(3).tolist()
        )
        if visual:

            def jitter(rgb):  # 明るさ(対数一様の倍率)と色味(R, G, B を別々に、一様)を変える
                return np.clip(rgb * logu(R["brightness_ratio"]) * (1 + u(R["hue"], 3)), 0, 1)

            m.geom_rgba[:] = b["geom_rgba"]
            for gid in range(m.ngeom):  # 材質のないもの(壁・パネルなど)は geom の色を変える
                if m.geom_matid[gid] < 0:
                    m.geom_rgba[gid, :3] = jitter(b["geom_rgba"][gid, :3])
            for mid in range(m.nmat):  # 材質のあるもの(ロボット)は材質の色を変える
                m.mat_rgba[mid, :3] = jitter(b["mat_rgba"][mid, :3])
            self._lcd_off = m.geom_rgba[self.lcd_geom].copy()
            li = float(logu(R["light_ratio"]))
            m.light_diffuse[:] = b["light_diffuse"] * li
            m.light_pos[:] = b["light_pos"] + u(R["light_pos_m"], b["light_pos"].shape)
            hl = float(logu(R["headlight_ratio"]))
            m.vis.headlight.diffuse[:] = b["headlight"] * hl
            params.update(light_intensity=round(li, 3), headlight=round(hl, 3))
            for name in ("top", "wrist"):
                cid = m.camera(name).id
                dp = g(R[f"{name}_cam_pos_m"], 3)
                rv = np.radians(g(R[f"{name}_cam_rot_deg"], 3))  # カメラの向きを小さく回す(カメラ座標で)
                dq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(dq, rv / (np.linalg.norm(rv) + 1e-12), np.linalg.norm(rv))
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, b["cam_quat"][cid], dq)
                df = g(R[f"{name}_cam_fovy_deg"])
                m.cam_pos[cid] = b["cam_pos"][cid] + dp
                m.cam_quat[cid] = q
                m.cam_fovy[cid] = b["cam_fovy"][cid] + df
                params[f"{name}_cam"] = dict(
                    dpos_mm=(dp * 1000).round(2).tolist(),
                    drot_deg=round(float(np.degrees(np.linalg.norm(rv))), 3),
                    dfovy_deg=round(float(df), 3),
                )
        mujoco.mj_forward(m, self.d)
        return params

    def restore_visual(self):
        """見た目(色・明るさ・照明)とカメラ(位置・向き・画角)を、較正した基準の値に戻す。
        randomize() のあとに呼ぶと、乱数は同じだけ引いたまま、見た目のばらつきだけを消せる
        (パネルの位置と動きは、呼ばない場合とまったく同じになる)。
        """
        m, b = self.m, self._base
        m.geom_rgba[:] = b["geom_rgba"]
        m.mat_rgba[:] = b["mat_rgba"]
        m.light_diffuse[:] = b["light_diffuse"]
        m.light_pos[:] = b["light_pos"]
        m.vis.headlight.diffuse[:] = b["headlight"]
        m.cam_pos[:] = b["cam_pos"]
        m.cam_quat[:] = b["cam_quat"]
        m.cam_fovy[:] = b["cam_fovy"]
        self._lcd_off = m.geom_rgba[self.lcd_geom].copy()
        mujoco.mj_forward(m, self.d)


class IK:
    """手先(gripperframe)の位置と向き(指先の方向)を合わせる減衰最小二乗法の IK。"""

    def __init__(self, sim):
        self.m = sim.m
        self.d = mujoco.MjData(self.m)
        self.qadr = sim.qadr
        self.site = self.m.site("gripperframe").id
        self.dofs = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in IK_JOINTS])
        self.lo = self.m.jnt_range[[self.m.joint(j).id for j in IK_JOINTS], 0]
        self.hi = self.m.jnt_range[[self.m.joint(j).id for j in IK_JOINTS], 1]

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3)[:, 0].copy()

    def solve(self, q_seed, pos, direction, iters=200, w_rot=0.05, lam=0.02):
        """pos: 指先の目標位置、direction: 指先が向く方向(単位ベクトル)。返り値: (q, 位置誤差[m], 角度誤差[rad])"""
        q = np.array(q_seed, float)
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        n = len(IK_JOINTS)
        for _ in range(iters):
            p, a = self.fk(q)
            mujoco.mj_comPos(self.m, self.d)
            ep = pos - p
            er = np.cross(a, direction)
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            P = np.eye(3) - np.outer(a, a)
            J = np.vstack([jp[:, self.dofs], w_rot * P @ jr[:, self.dofs]])
            e = np.concatenate([ep, w_rot * er])
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            dq = np.clip(dq, -0.2, 0.2)
            q[:n] = np.clip(q[:n] + dq, self.lo, self.hi)
        p, a = self.fk(q)
        return q, float(np.linalg.norm(pos - p)), float(math.acos(np.clip(a @ direction, -1, 1)))


class Units:
    """シミュレーションの関節角(ラジアン)と、LeRobot の値の変換。

    LeRobot 0.6.2 の so101_follower(use_degrees=True が既定)は、
      腕の5関節: キャリブレーションした可動範囲の中央を 0 とする「度」
      グリッパー: 可動範囲を 0〜100 にした値
    なので、既定は「度 = ラジアン × 180/π」と、グリッパーの範囲の 0〜100 対応にしてある。
    向きやずれが実機と違う関節は units.json で直す(check_alignment.py で確かめる)。
      {"elbow_flex": {"scale": -57.2958, "offset": 3.0}, ...}   実機の値 = scale × ラジアン + offset
    """

    def __init__(self, path="units.json", gripper_range=(-0.17453, 1.74533)):
        deg = 180.0 / math.pi
        g0, g1 = gripper_range
        self.scale = np.array([deg] * 5 + [100.0 / (g1 - g0)])
        self.offset = np.array([0.0] * 5 + [-g0 * 100.0 / (g1 - g0)])
        self.source = "既定値"
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                over = json.load(f)
            for i, j in enumerate(JOINTS):
                if j in over:
                    self.scale[i] = over[j].get("scale", self.scale[i])
                    self.offset[i] = over[j].get("offset", self.offset[i])
            self.source = os.path.abspath(path)

    def to_real(self, q):
        return (self.scale * np.asarray(q) + self.offset).astype(np.float32)

    def to_sim(self, v):
        return (np.asarray(v, float) - self.offset) / self.scale
