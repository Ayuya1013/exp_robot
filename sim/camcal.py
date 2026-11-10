r"""カメラの較正(内部パラメータ、ゆがみ、位置と向き)。ChArUco ボード(チェッカーボード + マーカー)を使う。

ボードは2種類: fine(13x9マス、15mm、手先用。既定)と big(7x5マス、35mm、俯瞰用)。どのコマンドにも同じ --board を付ける。
印刷が縮んでいてもよいが、1マスの実寸を測って --square [mm] で渡す(付けないと設計値で計算する)。

1) ボードを作る:      python camcal.py board --board fine
2) 撮る:              python camcal.py capture --board fine --square 13.3 --cam wrist --index 0 --out calib\wrist_int
                      (--port COM4 を付けると関節値も保存し、O・X・Y の印を爪で記録できる)
   キー: Space = 保存(交点が6個以上見えているときだけ)/ O・X・Y = 爪の先の位置を印として記録(何回か当てると平均)
         / R = 印の記録を消す / Q = 終了
3) 内部パラメータ:    python camcal.py intrinsics --board fine --square 13.3 --dir calib\wrist_int
4) 位置と向き:        python camcal.py pose --board fine --square 13.3 --cam wrist --dir calib\wrist_pose --intr calib\wrist_int\intrinsics.json
   cameras.json の該当カメラを書き換え、シミュと実画像を重ねた画像(overlay_*.png)を保存する。
5) 確認:              python camcal.py check --cam top --index 1      (俯瞰の実画像にシミュのパネルの輪郭を描く)
6) 色:                python camcal.py colors --cam top    (俯瞰の実画像から壁・パネル・液晶・ボタンの色を合わせる)
                      python camcal.py colors --cam wrist  (手先の実画像の爪から、ロボットの3Dプリント部品の色を合わせる)
"""
import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# ボードは2種類。どちらも A4 横に 100% で印刷する(縮んでも、マスの実寸を --square で渡せばよい)
#   big : 7 x 5 マス、35mm。俯瞰カメラ用(遠くから撮る)
#   fine: 13 x 9 マス、15mm。手先カメラ用(ピントが近いので小さいマスで、近くでも交点がたくさん写るようにする)
BOARDS = {
    "big": dict(squares=(7, 5), square_mm=35.0, marker_mm=26.0, dict=cv2.aruco.DICT_4X4_50),
    "fine": dict(squares=(13, 9), square_mm=15.0, marker_mm=11.0, dict=cv2.aruco.DICT_4X4_100),
}


def marks(name):
    """爪で触る交点の番号。O = 左上の内側の交点、X = 同じ行の右端、Y = 同じ列の下端。"""
    cols, rows = BOARDS[name]["squares"]
    return {"O": 0, "X": cols - 2, "Y": (rows - 2) * (cols - 1)}


def make_board(name="fine", square_mm=None):
    b = BOARDS[name]
    sq = square_mm or b["square_mm"]
    d = cv2.aruco.getPredefinedDictionary(b["dict"])
    return cv2.aruco.CharucoBoard(b["squares"], sq / 1000, b["marker_mm"] / b["square_mm"] * sq / 1000, d)


def _detector(board):
    cp = cv2.aruco.CharucoParameters()
    cp.tryRefineMarkers = True
    dp = cv2.aruco.DetectorParameters()
    dp.adaptiveThreshWinSizeMax = 53          # ぼけた画像やむらのある明るさでもマーカーを見つけやすくする
    dp.adaptiveThreshWinSizeStep = 4
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cp, dp)


def detect(board, img, min_corners=6):
    """交点を探す。そのまま、コントラストを上げたもの、2倍に拡大したもの、の順に試し、一番多く見つかったものを使う。
    返り値: (画像上の位置 Nx2, 番号 N) か None"""
    det = _detector(board)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    best = None
    for g, scale in ((gray, 1), (clahe, 1), (cv2.resize(clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2)):
        corners, ids, _, _ = det.detectBoard(g)
        if ids is not None and (best is None or len(ids) > len(best[1])):
            best = (corners.reshape(-1, 2) / scale, ids.reshape(-1))
    if best is None or len(best[1]) < min_corners:
        return None
    return best


# ---------------- board ----------------
def cmd_board(args):
    """A4 横(297 x 210mm)に 100% で印刷する PDF を作る。O・X・Y の印も付ける。"""
    from PIL import Image, ImageDraw, ImageFont
    dpi = 300
    px = lambda mm: int(round(mm / 25.4 * dpi))
    W, H = px(297), px(210)
    B = BOARDS[args.board]
    (cols, rows), sqmm = B["squares"], B["square_mm"]
    bw, bh = px(cols * sqmm), px(rows * sqmm)
    board = make_board(args.board)
    MARKS = marks(args.board)
    img = board.generateImage((bw, bh), marginSize=0)
    page = Image.new("L", (W, H), 255)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    page.paste(Image.fromarray(img), (x0, y0))
    draw = ImageDraw.Draw(page)
    obj = board.getChessboardCorners()
    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, px(5))
            break
        except OSError:
            pass
    font = font or ImageFont.load_default()

    def tri(x, y, d):  # 交点の方を指す三角(ボードの外の余白に描く)
        a = px(3)
        pts = {"down": [(x - a, y - 2 * a), (x + a, y - 2 * a), (x, y)], "right": [(x - 2 * a, y - a), (x - 2 * a, y + a), (x, y)]}[d]
        draw.polygon(pts, fill=0)

    # 印はボードの外に描く(交点の検出をじゃましないように)。プリンタの余白で切れないよう、紙の端から 8mm 以上離す
    for name, cid in MARKS.items():
        cx, cy = x0 + px(obj[cid][0] * 1000), y0 + px(obj[cid][1] * 1000)
        if name in ("O", "X"):  # 上の余白から、真下の交点を指す
            tri(cx, y0 - px(0.5), "down")
            draw.text((cx + px(4), y0 - px(8)), name, fill=0, font=font)
        if name in ("O", "Y"):  # 左の余白から、真右の交点を指す
            tri(x0 - px(0.5), cy, "right")
            draw.text((x0 - px(14), cy - px(8)), name, fill=0, font=font)
    small = ImageFont.truetype(font.path, px(3)) if hasattr(font, "path") else font
    draw.text((x0, y0 + bh + px(2)), f"ChArUco '{args.board}' {cols}x{rows}  square {sqmm}mm  marker {B['marker_mm']}mm. "
              "Print at 100% and measure one square.", fill=0, font=small)
    draw.text((x0, y0 + bh + px(6)), "O = inner corner 1 square in from the top-left corner.  "
              f"X = {cols - 2} squares right of O.  Y = {rows - 2} squares below O.", fill=0, font=small)
    out = args.out or f"charuco_{args.board}_A4.pdf"
    page.save(out, resolution=dpi)
    print(f"作成: {os.path.abspath(out)}  (A4 横、拡大縮小なしで印刷。印 O・X・Y は交点の位置を示す)")


# ---------------- capture ----------------
def cmd_capture(args):
    import tkinter as tk
    from PIL import Image, ImageTk
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    os.makedirs(args.out, exist_ok=True)
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.index, width=640, height=480, fps=30, fourcc="MJPG"))
    cam.connect()
    board = make_board(args.board, args.square)

    robot = sim = units = None
    if args.port:
        import check_alignment as C
        import so101_sim as S
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
        robot.connect(calibrate=False)
        robot.bus.disable_torque()
        sim = S.Sim(args.model_dir, render=False)
        units = S.Units(os.path.join(HERE, "units.json"))
        print("モーターの力を抜きました。腕を手で支えてください。")

    def joints():
        pos = robot.bus.sync_read("Present_Position")
        return [float(pos[j]) for j in S.JOINTS]

    pts_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"board_points_{args.board}.json")
    points = json.load(open(pts_path, encoding="utf-8")) if os.path.exists(pts_path) else {}
    n = len([f for f in glob.glob(os.path.join(args.out, f"{args.cam}_*.png"))])
    root = tk.Tk()
    root.title(f"capture {args.cam}  (Space 保存 / H 姿勢を保持・解除 / O X Y 印を記録(平均を使う)/ R 印を消す / Q 終了)")
    label = tk.Label(root)
    label.pack()
    status = tk.Label(root, font=("", 12))
    status.pack()
    state = {"img": None, "found": 0, "hold": False}

    def key(e):
        nonlocal n
        k = e.keysym.lower()
        if k == "q":
            root.quit()
        elif k == "space" and state["img"] is not None:
            if state["found"] < 6:
                print(f"交点が {state['found']} 個しか見えていないので保存しません(6個以上必要。ボードを近づける、ピントや明るさを確かめる)")
                return
            base = os.path.join(args.out, f"{args.cam}_{n:03d}")
            img = state["img"]
            if robot is not None:
                # 腕が止まっているかを確かめ、カメラのたまった古いコマを捨ててから、関節値と同じ瞬間の画像を撮る
                j1 = np.mean([joints() for _ in range(5)], axis=0)
                for _ in range(4):
                    img = cam.read()
                j2 = np.mean([joints() for _ in range(5)], axis=0)
                moved = np.abs(j2 - j1)[:5].max()
                if moved > 0.3:
                    print(f"腕が動いています({moved:.1f}度)。止めてからもう一度(H で姿勢を保持できる)")
                    return
                json.dump({"joints_real": ((j1 + j2) / 2).tolist(), "held": state["hold"]}, open(base + ".json", "w"))
            Image.fromarray(img).save(base + ".png")
            n += 1
            print(f"保存: {base}.png" + ("(保持中)" if state["hold"] else ""))
        elif k == "h" and robot is not None:
            # 保持: 今の姿勢でモーターに力を入れる(実際の動作と同じく、重さをモーターが支えた状態で測れる)
            if not state["hold"]:
                pos = robot.bus.sync_read("Present_Position")
                robot.bus.sync_write("Goal_Position", pos)
                robot.bus.enable_torque()
                state["hold"] = True
                print("今の姿勢で保持しました(もう一度 H で力を抜く)")
            else:
                robot.bus.disable_torque()
                state["hold"] = False
                print("力を抜きました。腕を手で支えてください")
        elif k in ("o", "x", "y") and robot is not None:
            real = np.mean([joints() for _ in range(10)], axis=0)
            tip = C.fingertip_world(sim, units.to_sim(real))
            samples = points.get(k.upper(), [])
            if samples and not isinstance(samples[0], list):
                samples = [samples]
            samples.append(tip.tolist())  # 何回か当て直すと、その平均を使う
            points[k.upper()] = samples
            json.dump(points, open(pts_path, "w"), indent=2)
            arr = np.array(samples) * 1000
            print(f"印 {k.upper()}: {np.round(tip * 100, 1)} cm  ({len(samples)} 回、平均 {np.round(arr.mean(0) / 10, 1)} cm、"
                  f"ばらつき {arr.std(0).max() if len(arr) > 1 else 0:.1f} mm)")
        elif k == "r" and robot is not None:
            points.clear()
            json.dump(points, open(pts_path, "w"), indent=2)
            print("印の記録を消しました")

    def loop():
        frame = cam.read()  # RGB、LeRobot と同じ
        state["img"] = frame
        show = frame.copy()
        r = detect(board, frame, min_corners=1)
        state["found"] = 0 if r is None else len(r[1])
        if r is not None:
            for p in r[0]:
                cv2.circle(show, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        photo = ImageTk.PhotoImage(Image.fromarray(show))
        label.configure(image=photo)
        label.image = photo
        marks = " ".join(f"{k}x{len(v) if isinstance(v[0], list) else 1}" for k, v in sorted(points.items()))
        status.configure(text=f"保存 {n} 枚  交点 {state['found']}  印 {marks or 'なし'}" + ("  [保持中]" if state["hold"] else ""))
        root.after(30, loop)

    root.bind("<Key>", key)
    loop()
    root.mainloop()
    cam.disconnect()
    if robot is not None:
        robot.disconnect()


# ---------------- diag ----------------
def cmd_diag(args):
    """交点が出ないときの切り分け。1枚撮って(または画像を読んで)、どのボード・向きならマーカーと交点が見つかるかを調べる。"""
    if args.image:
        img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    else:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.index, width=640, height=480, fps=30, fourcc="MJPG"))
        cam.connect()
        for _ in range(10):
            img = cam.read()
        cam.disconnect()
    cv2.imwrite("diag.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f"画像 {img.shape[1]}x{img.shape[0]}  明るさ 平均 {gray.mean():.0f}  最小 {gray.min()}  最大 {gray.max()}(diag.png に保存)")
    show = img.copy()
    for flip_name, g in (("そのまま", gray), ("左右反転", gray[:, ::-1].copy())):
        for name in BOARDS:
            board = make_board(name)
            d = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(BOARDS[name]["dict"]))
            mk, ids, _ = d.detectMarkers(g)
            r = detect(board, g, min_corners=1)
            n_ids = 0 if ids is None else len(ids)
            print(f"  {flip_name:5s} ボード {name:4s}: マーカー {n_ids:3d} 個(番号 {sorted(ids.ravel().tolist())[:8] if n_ids else '-'}…)"
                  f"  交点 {0 if r is None else len(r[1])} 個")
            if flip_name == "そのまま" and n_ids:
                cv2.aruco.drawDetectedMarkers(show, mk, ids)
    cv2.imwrite("diag_markers.png", cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
    print("diag_markers.png に、見つかったマーカーを描いた画像を保存。")
    print("見方: マーカーが0なら、写り方(遠い・ぼけ・反射・暗い)。左右反転だけで見つかるなら、カメラの映像が鏡像になっている。"
          "マーカーはあるのに交点が0なら、--board の指定が印刷したボードと違う。")


# ---------------- check ----------------
def project_boxes(sim, cam, names):
    """箱の geom の正面(パネル座標の -y 側)の四隅を、較正したカメラ(ゆがみ込み)で画像に写す。返り値: {名前: 4x2}"""
    import mujoco
    import env_builder as E
    m, d = sim.m, sim.d
    mujoco.mj_forward(m, d)
    cid = m.camera(cam).id
    R_cv = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # OpenCV のカメラ: x 右、y 下、z 前
    pc = d.cam_xpos[cid]
    ccfg = sim.env["cameras"][cam]
    if ccfg.get("intrinsics"):
        path = ccfg["intrinsics"] if os.path.isabs(ccfg["intrinsics"]) else os.path.join(E.HERE, ccfg["intrinsics"])
        intr = json.load(open(path, encoding="utf-8"))
        K, dist = np.array(intr["pinhole"]["K"]), np.array(intr["pinhole"]["dist"])
    else:
        h = sim.cfg["img_h"]
        f = h / 2 / math.tan(math.radians(m.cam_fovy[cid]) / 2)
        K, dist = np.array([[f, 0, sim.cfg["img_w"] / 2], [0, f, h / 2], [0, 0, 1]]), np.zeros(5)
    out = {}
    for name in names:
        g = m.geom(name).id
        sx, sy, sz = m.geom_size[g]
        Rg = d.geom_xmat[g].reshape(3, 3)
        loc = np.array([[-sx, -sy, sz], [sx, -sy, sz], [sx, -sy, -sz], [-sx, -sy, -sz]])
        world = d.geom_xpos[g] + loc @ Rg.T
        cam_pts = (world - pc) @ R_cv
        p, _ = cv2.projectPoints(cam_pts.astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
        out[name] = p.reshape(-1, 2)
    return out


def project_world(sim, cam, pts):
    """ロボット座標の点を、較正したカメラ(ゆがみ込み)で画像に写す。"""
    import mujoco
    import env_builder as E
    m, d = sim.m, sim.d
    mujoco.mj_forward(m, d)
    cid = m.camera(cam).id
    R_cv = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    ccfg = sim.env["cameras"][cam]
    if ccfg.get("intrinsics"):
        path = ccfg["intrinsics"] if os.path.isabs(ccfg["intrinsics"]) else os.path.join(E.HERE, ccfg["intrinsics"])
        intr = json.load(open(path, encoding="utf-8"))
        K, dist = np.array(intr["pinhole"]["K"]), np.array(intr["pinhole"]["dist"])
    else:
        h = sim.cfg["img_h"]
        f = h / 2 / math.tan(math.radians(m.cam_fovy[cid]) / 2)
        K, dist = np.array([[f, 0, sim.cfg["img_w"] / 2], [0, f, h / 2], [0, 0, 1]]), np.zeros(5)
    cam_pts = (np.asarray(pts, float) - d.cam_xpos[cid]) @ R_cv
    p, _ = cv2.projectPoints(cam_pts, np.zeros(3), np.zeros(3), K, dist)
    return p.reshape(-1, 2)


def color_rois(sim):
    """色を測る場所(名前 → (色を合わせる geom, ロボット座標の四角形 4x3))。パネル座標(x 右、y 奥、z 上)で決める。"""
    import mujoco
    m, d = sim.m, sim.d
    mujoco.mj_forward(m, d)
    Rm, pm = d.xmat[sim.mount].reshape(3, 3), d.xpos[sim.mount]
    to_world = lambda loc: pm + np.asarray(loc) @ Rm.T
    inv = lambda g: Rm.T @ (d.geom_xpos[m.geom(g).id] - pm)  # geom の中心(パネル座標)

    def front_quad(x0, x1, z0, z1, y):
        return to_world([[x0, y, z1], [x1, y, z1], [x1, y, z0], [x0, y, z0]])

    def box_front(g, shrink):
        c, (sx, sy, sz) = inv(g), m.geom_size[m.geom(g).id]
        return front_quad(c[0] - sx * shrink, c[0] + sx * shrink, c[2] - sz * shrink, c[2] + sz * shrink, c[1] - sy - 0.0005)

    wall_c, wall_s = inv("wall"), m.geom_size[m.geom("wall").id]
    wall_y = wall_c[1] - wall_s[1] - 0.0005
    rois = {
        "panel": (["panel/panel_board"], front_quad(-0.056, -0.040, 0.005, 0.045, -0.0105)),  # 液晶の左の余白
        "lcd": (["panel/lcd_screen"], box_front("panel/lcd_screen", 0.7)),
        "buttons": ([f"panel/geom_{b}" for b in ("mode", "power", "fan", "cancel")], None),
        "wall": (["wall"], front_quad(-0.13, -0.075, -0.04, 0.04, wall_y)),  # パネルの左の壁
    }
    rois["buttons"] = (rois["buttons"][0], [box_front(g, 0.6) for g in rois["buttons"][0]])
    if "side_wall" in [m.geom(i).name for i in range(m.ngeom)]:
        sc, ss = inv("side_wall"), m.geom_size[m.geom("side_wall").id]
        x = sc[0] - ss[0] - 0.0005
        y0, y1 = sc[1] + ss[1] * 0.2, sc[1] + ss[1] * 0.8  # パネルの壁に近い側
        rois["side_wall"] = (["side_wall"], to_world([[x, y0, 0.04], [x, y1, 0.04], [x, y1, -0.04], [x, y0, -0.04]]))
    return rois


def roi_mean(img, polys):
    mask = np.zeros(img.shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], 1)
    if mask.sum() < 30:
        return None, mask
    return img[mask > 0].reshape(-1, 3).mean(axis=0), mask


def seg_mask(sim, cam, geom_ids):
    """シミュのカメラ画像で、指定した geom が写っている画素(ゆがみ込み)。"""
    import mujoco
    lens = sim.env["cameras"][cam].get("_lens")
    h, w = sim.cfg["img_h"], sim.cfg["img_w"]
    r = mujoco.Renderer(sim.m, lens["H"] if lens else h, lens["W"] if lens else w)
    r.enable_segmentation_rendering()
    r.update_scene(sim.d, camera=cam)
    seg = r.render()[..., 0].astype(np.int32)  # geom の番号(なければ -1)
    r.close()
    if lens:
        seg = cv2.remap(seg.astype(np.float32), lens["mapx"], lens["mapy"], cv2.INTER_NEAREST, borderValue=-1).astype(np.int32)
    mask = np.isin(seg, list(geom_ids)).astype(np.uint8)
    return cv2.erode(mask, np.ones((9, 9), np.uint8))  # 縁は較正のずれで外れるので削る


def grab(args):
    if args.image:
        return cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.index, width=640, height=480, fps=30, fourcc="MJPG"))
    cam.connect()
    for _ in range(20):
        img = cam.read()
    cam.disconnect()
    return img


def save_colors(path, updates, report_key, report):
    colors = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            colors = json.load(f)
    colors.update(updates)
    colors.setdefault("_measured", {})[report_key] = report
    with open(path, "w", encoding="utf-8") as f:
        json.dump(colors, f, indent=2, ensure_ascii=False)


def cmd_colors(args):
    """実画像から色を測り、シミュの描画が同じ色になるよう基準の色を合わせて colors.json に書く。
    --cam top  : 俯瞰の画像で、壁・右の壁・パネル・液晶・ボタン
    --cam wrist: 手先の画像で、ロボットの3Dプリント部品(画像の下に写る固定側の爪で測る。爪はカメラに対して動かない)"""
    import imageio.v2 as imageio
    import so101_sim as S
    if args.index is None:
        args.index = 1 if args.cam == "top" else 0
    real = grab(args)
    out_path = os.path.join(HERE, "colors.json")
    sim = S.Sim(args.model_dir, theta=0.0)
    q = np.array(args.home)
    if args.port:  # 今の腕の姿勢をシミュに反映する(照明の当たり方をそろえるため)
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
        robot.connect(calibrate=False)
        pos = robot.bus.sync_read("Present_Position")
        robot.disconnect()
        q = S.Units(os.path.join(HERE, "units.json")).to_sim([pos[j] for j in S.JOINTS])
    sim.reset(q)
    os.makedirs("calib", exist_ok=True)

    if args.cam == "wrist":
        m = sim.m
        jaw = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("gripper").id and m.geom_matid[g] >= 0
               and m.geom_matid[g] in S.robot_part_materials(m)]
        mask = seg_mask(sim, "wrist", jaw)
        if mask.sum() < 50:
            raise SystemExit("手先の画像に固定側の爪がほとんど写っていません")
        real_mean = real[mask > 0].reshape(-1, 3).mean(axis=0)
        mats = S.robot_part_materials(m)
        before = sim.render()["wrist"]
        for _ in range(4):
            sm = sim.render()["wrist"][mask > 0].reshape(-1, 3).mean(axis=0)
            ratio = (real_mean + 1) / (sm + 1)
            for i in mats:
                m.mat_rgba[i, :3] = np.clip(m.mat_rgba[i, :3] * ratio, 0.02, 1.0)
        after = sim.render()["wrist"]
        sm = after[mask > 0].reshape(-1, 3).mean(axis=0)
        rgb = [round(float(v), 4) for v in m.mat_rgba[mats[0], :3]]
        print(f"  爪(3Dプリント部品) 実画像 {real_mean.round(0)}  シミュ {sm.round(0)}  → 基準の色 {rgb}"
              + ("  ※色が上限(1.0)に達した。照明が暗い" if max(rgb) >= 0.999 else ""))
        save_colors(out_path, {"robot_parts": rgb}, "robot_parts", dict(real=real_mean.round(1).tolist(), sim=sm.round(1).tolist()))
        show = real.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(show, cnts, -1, (255, 255, 0), 1)
        imageio.imwrite(os.path.join("calib", "colors_compare_wrist.png"), np.concatenate([show, before, after], axis=1)[::2, ::2])
        print(f"保存: {out_path}(robot_parts)")
        print("保存: calib\\colors_compare_wrist.png(左: 実画像と測った場所、中: 合わせる前、右: 合わせた後)")
        return

    rois = color_rois(sim)
    polys = {k: ([v[1]] if not isinstance(v[1], list) else v[1]) for k, v in rois.items()}
    pix = {k: [project_world(sim, "top", q) for q in qs] for k, qs in polys.items()}
    real_mean = {k: roi_mean(real, p)[0] for k, p in pix.items()}
    before = sim.render()["top"]
    for it in range(4):  # 描画して比べ、色の倍率を直す、をくり返す(照明の当たり方も込みで合う)
        simg = sim.render()["top"]
        for k, (geoms, _) in rois.items():
            if real_mean[k] is None:
                continue
            sm, _ = roi_mean(simg, pix[k])
            ratio = (real_mean[k] + 1) / (sm + 1)
            for g in geoms:
                gid = sim.m.geom(g).id
                sim.m.geom_rgba[gid, :3] = np.clip(sim.m.geom_rgba[gid, :3] * ratio, 0.02, 1.0)
    after = sim.render()["top"]
    colors, report = {}, {}
    for k, (geoms, _) in rois.items():
        if real_mean[k] is None:
            print(f"  {k}: 画像に写っていないので測れません")
            continue
        sm, _ = roi_mean(after, pix[k])
        for g in geoms:
            colors[g] = [round(float(v), 4) for v in sim.m.geom_rgba[sim.m.geom(g).id, :3]]
        sat = any(v >= 0.999 for g in geoms for v in colors[g])
        report[k] = dict(real=real_mean[k].round(1).tolist(), sim=sm.round(1).tolist())
        print(f"  {k:9s} 実画像 {real_mean[k].round(0)}  シミュ {sm.round(0)}" + ("  ※色が上限(1.0)に達した。照明が暗い" if sat else ""))
    save_colors(out_path, colors, "top", report)
    show = real.copy()
    for k, ps in pix.items():
        for p in ps:
            cv2.polylines(show, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 1)
    imageio.imwrite(os.path.join("calib", "colors_compare.png"), np.concatenate([show, before, after], axis=1)[::2, ::2])
    print(f"保存: {out_path}(シミュはこれを基準の色にする)")
    print("保存: calib\\colors_compare.png(左: 実画像と測った場所、中: 合わせる前、右: 合わせた後)")


def cmd_check(args):
    """今のカメラの設定で、パネル・液晶・運転/停止の輪郭を実画像に描いて、ずれを目で確かめる(ボードは不要)。"""
    import imageio.v2 as imageio
    import so101_sim as S
    if args.image:
        real = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    else:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.index, width=640, height=480, fps=30, fourcc="MJPG"))
        cam.connect()
        for _ in range(15):  # 露出とピントが落ち着くまで捨てる
            real = cam.read()
        cam.disconnect()
    sim = S.Sim(args.model_dir)
    q = np.array(args.home)
    if args.port:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
        robot.connect(calibrate=False)
        pos = robot.bus.sync_read("Present_Position")
        robot.disconnect()
        q = S.Units(os.path.join(HERE, "units.json")).to_sim([pos[j] for j in S.JOINTS])
    sim.reset(q)
    simg = sim.render()[args.cam]
    boxes = project_boxes(sim, args.cam, ["panel/panel_board", "panel/lcd_screen", "panel/geom_power"])
    draw = real.copy()
    colors = {"panel/panel_board": (255, 0, 0), "panel/lcd_screen": (0, 255, 255), "panel/geom_power": (0, 255, 0)}
    for name, pts in boxes.items():
        cv2.polylines(draw, [pts.astype(np.int32).reshape(-1, 1, 2)], True, colors[name], 2)
    cv2.putText(draw, "red: panel  cyan: LCD  green: power (sim)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    os.makedirs("calib", exist_ok=True)
    out1 = os.path.join("calib", f"check_{args.cam}_outline.png")
    out2 = os.path.join("calib", f"check_{args.cam}_blend.png")
    imageio.imwrite(out1, draw)
    imageio.imwrite(out2, (0.5 * real + 0.5 * simg).astype(np.uint8))
    print(f"保存: {out1}(実画像に、シミュのパネル=赤、液晶=水色、運転/停止=緑 の輪郭)")
    print(f"保存: {out2}(実画像とシミュの画像を半分ずつ重ねたもの)")
    for name, pts in boxes.items():
        print(f"  {name.split('/')[-1]:12s} 画像上の範囲 x {pts[:, 0].min():.0f}〜{pts[:, 0].max():.0f}  y {pts[:, 1].min():.0f}〜{pts[:, 1].max():.0f}")


# ---------------- intrinsics ----------------
def load_detections(board, folder):
    obj_all = board.getChessboardCorners()
    objs, imgs, names, size = [], [], [], None
    for f in sorted(glob.glob(os.path.join(folder, "*.png"))):
        if os.path.basename(f).startswith(("overlay_", "check_")):  # このツールが作った画像は使わない
            continue
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        size = (img.shape[1], img.shape[0])
        r = detect(board, img)
        if r is None:
            print(f"  交点が足りないので使わない: {os.path.basename(f)}")
            continue
        imgs.append(r[0].astype(np.float32))
        objs.append(obj_all[r[1]].astype(np.float32))
        names.append(f)
    return objs, imgs, names, size


def distortion_px(K, dist, size, fisheye=False):
    """画像の四隅と辺の中点で、ゆがみによって何ピクセルずれているか(最大値)。"""
    w, h = size
    pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, 0], [0, h / 2]], np.float32).reshape(-1, 1, 2)
    if fisheye:
        und = cv2.fisheye.undistortPoints(pts, K, dist)
    else:
        und = cv2.undistortPoints(pts, K, dist)
    ideal = und.reshape(-1, 2) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return float(np.max(np.linalg.norm(ideal - pts.reshape(-1, 2), axis=1)))


def per_image_errors(objs, imgs, K, dist, rvecs, tvecs):
    """画像ごと・交点ごとの再投影誤差 [px]。"""
    out = []
    for o, i, r, t in zip(objs, imgs, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        out.append(np.linalg.norm(p.reshape(-1, 2) - i.reshape(-1, 2), axis=1))
    return out


def calibrate_robust(objs, imgs, size, flags=0, K0=None, rounds=4):
    """外れた交点と画像を除きながら、くり返し較正する。返り値: (rms, K, dist, 残した objs, imgs, 番号, 交点の除外数)"""
    keep = list(range(len(objs)))
    o, i = [x.copy() for x in objs], [x.copy() for x in imgs]
    dropped = 0
    for _ in range(rounds):
        oo, ii = [o[k] for k in keep], [i[k] for k in keep]
        rms, K, dist, rv, tv = cv2.calibrateCamera(oo, ii, size, None if K0 is None else K0.copy(), None, flags=flags)
        errs = per_image_errors(oo, ii, K, dist, rv, tv)
        med = np.median(np.concatenate(errs))
        thr = max(1.0, 4 * med)
        changed = False
        new_keep = []
        for k, e in zip(keep, errs):
            good = e < thr
            if good.sum() < len(e):
                dropped += int((~good).sum())
                o[k], i[k] = o[k][good], i[k][good]
                changed = True
            if len(o[k]) >= 6:
                new_keep.append(k)
            else:
                changed = True
        keep = new_keep
        if not changed:
            break
    oo, ii = [o[k] for k in keep], [i[k] for k in keep]
    rms, K, dist, rv, tv = cv2.calibrateCamera(oo, ii, size, None if K0 is None else K0.copy(), None, flags=flags)
    return rms, K, dist, oo, ii, keep, dropped, per_image_errors(oo, ii, K, dist, rv, tv)


def coverage_map(imgs, size, nx=8, ny=6):
    w, h = size
    grid = np.zeros((ny, nx), int)
    for i in imgs:
        for x, y in i.reshape(-1, 2):
            grid[min(ny - 1, int(y / h * ny)), min(nx - 1, int(x / w * nx))] += 1
    return grid


def cmd_intrinsics(args):
    board = make_board(args.board, args.square)
    objs, imgs, names, size = load_detections(board, args.dir)
    if len(objs) < 8:
        raise SystemExit(f"使える画像が {len(objs)} 枚しかありません(8枚以上必要)")
    w, h = size
    # 0) そのまま(参考)
    rms0, K0_, d0_, rv0, tv0 = cv2.calibrateCamera(objs, imgs, size, None, None)
    e0 = per_image_errors(objs, imgs, K0_, d0_, rv0, tv0)
    # A) 外れ値を除いて、主点も自由に求める
    A = calibrate_robust(objs, imgs, size)
    # B) 主点を画像の中心に固定し、k3 も 0 にする(画像の一部にしか交点がないときに安定する)
    Kc = np.array([[A[1][0, 0], 0, w / 2], [0, A[1][1, 1], h / 2], [0, 0, 1]])
    B = calibrate_robust(objs, imgs, size, K0=Kc,
                         flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3)

    def summary(name, R):
        rms, K, dist, oo, ii, keep, dropped, errs = R
        return dict(name=name, rms=rms, K=K.tolist(), dist=dist.ravel().tolist(), images=len(keep), dropped_corners=dropped,
                    fovy_deg=math.degrees(2 * math.atan(h / 2 / K[1, 1])), fovx_deg=math.degrees(2 * math.atan(w / 2 / K[0, 0])),
                    distortion_px=distortion_px(K, dist, size))

    # C) 主点は自由、k3 だけ 0(画像の隅に交点が少ないとき、隅のゆがみを大きく外挿しすぎないようにする)
    Cm = calibrate_robust(objs, imgs, size, flags=cv2.CALIB_FIX_K3)
    sA, sB, sC = summary("free", A), summary("fixed_center", B), summary("no_k3", Cm)
    off = math.hypot(A[1][0, 2] - w / 2, A[1][1, 2] - h / 2)
    if off >= 40 or sA["rms"] >= 1.0:
        use = sB  # 主点が大きくずれたら、中心に固定したほうを使う
    else:
        use = sC if sC["rms"] <= sA["rms"] + 0.05 else sA  # 誤差がほぼ同じなら、少ない係数のほう
    out = dict(size=[w, h], board=args.board, square_mm=args.square, use=use["name"], models=[sA, sC, sB],
               pinhole=dict(rms=use["rms"], K=use["K"], dist=use["dist"], distortion_px=use["distortion_px"]),
               fovy_deg=use["fovy_deg"], fovx_deg=use["fovx_deg"])
    path = os.path.join(args.dir, "intrinsics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n画像 {len(objs)} 枚  大きさ {w}x{h}")
    print(f"そのまま:                 再投影誤差 {rms0:.3f}px  主点 ({K0_[0, 2]:.0f}, {K0_[1, 2]:.0f})")
    worst = sorted(zip([float(np.sqrt(np.mean(e ** 2))) for e in e0], names), reverse=True)[:5]
    print("  誤差の大きい画像: " + ", ".join(f"{os.path.basename(n)} {v:.1f}px" for v, n in worst))
    names_ = {"free": "外れ値を除く", "no_k3": "外れ値を除く+k3なし", "fixed_center": "外れ値を除く+主点を中心に固定+k3なし"}
    for s_ in (sA, sC, sB):
        K = np.array(s_["K"])
        label = names_[s_["name"]]
        print(f"{label}: 再投影誤差 {s_['rms']:.3f}px  画像 {s_['images']} 枚  除いた交点 {s_['dropped_corners']}  "
              f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f}  主点 ({K[0, 2]:.0f}, {K[1, 2]:.0f})")
        print(f"    画角 縦 {s_['fovy_deg']:.1f}度 横 {s_['fovx_deg']:.1f}度  ゆがみ(四隅で) 最大 {s_['distortion_px']:.0f}px")
    print(f"→ 使うのは「{names_[use['name']]}」(ゆがみは画像の隅ほど不確か。交点の少ない隅の値は外挿)")
    print("交点が写った場所(画像を 8x6 に分けたときの交点の数。0 の所は較正の根拠がない):")
    for row in coverage_map(A[4], size):
        print("   " + " ".join(f"{v:4d}" for v in row))
    print(f"保存: {os.path.abspath(path)}")


# ---------------- pose ----------------
def board_in_base(board, points, MARKS):
    """爪で記録した印 O・X・Y(ロボット座標)から、ボード座標 → ロボット座標の変換(R, t)を求める。"""
    obj = board.getChessboardCorners()
    P = {}
    for k in ("O", "X", "Y"):
        v = np.array(points[k], float)
        P[k] = v.mean(axis=0) if v.ndim == 2 else v  # 何回か記録したら平均

    ob = {k: obj[MARKS[k]].astype(float) for k in MARKS}

    def basis(o, x, y):
        ex = (x - o) / np.linalg.norm(x - o)
        ey = (y - o) - ((y - o) @ ex) * ex
        ey /= np.linalg.norm(ey)
        return np.column_stack([ex, ey, np.cross(ex, ey)])

    R = basis(P["O"], P["X"], P["Y"]) @ basis(ob["O"], ob["X"], ob["Y"]).T
    t = P["O"] - R @ ob["O"]
    # 確認: 測った長さとボードの長さ
    for a, b in (("O", "X"), ("O", "Y")):
        m_, b_ = np.linalg.norm(P[a] - P[b]) * 1000, np.linalg.norm(ob[a] - ob[b]) * 1000
        print(f"  印 {a}-{b}: 爪で {m_:.1f}mm / ボード {b_:.1f}mm(比 {m_ / b_:.3f})")
    return R, t


def rot_to_quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def avg_rotation(Rs):
    """回転の平均(四元数の平均、向きをそろえてから)。"""
    qs = [rot_to_quat(R) for R in Rs]
    q0 = qs[0]
    qs = [q if q @ q0 >= 0 else -q for q in qs]
    q = np.mean(qs, axis=0)
    q /= np.linalg.norm(q)
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def cmd_pose(args):
    import env_builder as E
    import so101_sim as S
    board = make_board(args.board, args.square)
    if not os.path.exists(args.intr):
        found = glob.glob(os.path.join("calib", "**", "intrinsics.json"), recursive=True)
        raise SystemExit(f"{os.path.abspath(args.intr)} がありません。見つかった intrinsics.json: {found or 'なし'}")
    intr = json.load(open(args.intr, encoding="utf-8"))
    K = np.array(intr["pinhole"]["K"])
    dist = np.array(intr["pinhole"]["dist"])
    pts_path = args.points or os.path.join(os.path.dirname(os.path.abspath(args.dir)), f"board_points_{args.board}.json")
    points = json.load(open(pts_path, encoding="utf-8"))
    R_bb, t_bb = board_in_base(board, points, marks(args.board))  # ボード → ロボット
    obj_all = board.getChessboardCorners()
    sim = S.Sim(args.model_dir, render=False)
    units = S.Units(os.path.join(HERE, "units.json"))
    gid = sim.m.body("gripper").id

    Rs, ts, used = [], [], []
    for f in sorted(glob.glob(os.path.join(args.dir, "*.png"))):
        if os.path.basename(f).startswith(("overlay_", "check_")):  # このツールが作った画像は使わない
            continue
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        r = detect(board, img)
        if r is None:
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj_all[r[1]].astype(np.float32), r[0].astype(np.float32), K, dist,
                                                 reprojectionError=3.0, iterationsCount=200)
        if not ok or inl is None or len(inl) < 6:
            print(f"  ボードの向きを求められないので使わない: {os.path.basename(f)}")
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj_all[r[1]][inl].astype(np.float32), r[0][inl].astype(np.float32), K, dist, rvec, tvec)
        r = (r[0][inl], r[1][inl])  # 外れた交点は以後も使わない
        R_cb, _ = cv2.Rodrigues(rvec)  # ボード → カメラ(OpenCV: x 右、y 下、z 前)
        # カメラ → ロボット
        R_bc = R_bb @ R_cb.T
        t_bc = t_bb - R_bc @ tvec.ravel()
        if args.cam == "wrist":  # グリッパー基準に直す
            js = os.path.splitext(f)[0] + ".json"
            if not os.path.exists(js):
                print(f"  関節値がないので使わない: {os.path.basename(f)}")
                continue
            q = units.to_sim(json.load(open(js))["joints_real"])
            sim.d.qpos[sim.qadr] = q
            import mujoco
            mujoco.mj_kinematics(sim.m, sim.d)
            Rg, pg = sim.d.xmat[gid].reshape(3, 3), sim.d.xpos[gid]
            R_bc, t_bc = Rg.T @ R_bc, Rg.T @ (t_bc - pg)
        Rs.append(R_bc)
        ts.append(t_bc)
        used.append((f, r))
    if not Rs:
        raise SystemExit("ボードが写った画像がありません")
    t = np.mean(ts, axis=0)
    R = avg_rotation(Rs)
    spread = np.std(ts, axis=0) * 1000 if len(ts) > 1 else np.zeros(3)
    ang = [math.degrees(math.acos(np.clip((np.trace(R.T @ Ri) - 1) / 2, -1, 1))) for Ri in Rs]
    R0, t0 = R.copy(), t.copy()
    frame = "グリッパー" if args.cam == "wrist" else "ロボットの台"
    print(f"\n{args.cam}: 画像 {len(Rs)} 枚。カメラの位置({frame}基準)[{t[0] * 100:.1f}, {t[1] * 100:.1f}, {t[2] * 100:.1f}] cm")
    print(f"  ばらつき: 位置 {spread.round(1)} mm、向き 最大 {max(ang):.1f}度")

    import mujoco
    data = []
    for f, (pts, ids) in used:
        real = json.load(open(os.path.splitext(f)[0] + ".json"))["joints_real"] if args.cam == "wrist" else None
        data.append((f, pts, ids, real))

    def predict(Rc, tc, Rb, tb, delta, item):
        """カメラ(グリッパー基準 or 台基準)、ボード、関節のずれ delta [rad] から、交点が写る位置を計算する。"""
        f, pts, ids, real = item
        if real is not None:
            q = units.to_sim(real)
            q[:5] += delta
            sim.d.qpos[sim.qadr] = q
            mujoco.mj_kinematics(sim.m, sim.d)
            Rg, pg = sim.d.xmat[gid].reshape(3, 3), sim.d.xpos[gid]
            Rw, tw = Rg @ Rc, pg + Rg @ tc
        else:
            Rw, tw = Rc, tc
        P = (Rb @ obj_all[ids].T).T + tb
        pc = (Rw.T @ (P - tw).T).T
        proj, _ = cv2.projectPoints(pc.astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
        return proj.reshape(-1, 2)

    def evaluate(Rc, tc, Rb, tb, delta, label, save=False):
        """全部つなげた確認: 爪で測ったボードの位置 + 順運動学 + カメラ + レンズで、交点が写るはずの位置と実際を比べる。"""
        allv = []
        for item in data:
            f, pts, ids, _ = item
            proj = predict(Rc, tc, Rb, tb, delta, item)
            e = np.linalg.norm(proj - pts, axis=1)
            allv.append(e)
            if save:
                img = cv2.imread(f)
                for p1, p2 in zip(pts, proj):
                    cv2.circle(img, (int(p1[0]), int(p1[1])), 3, (0, 200, 0), -1)   # 緑: 見つかった交点
                    cv2.circle(img, (int(p2[0]), int(p2[1])), 5, (0, 0, 255), 1)    # 赤い輪: 計算で写るはずの位置
                cv2.putText(img, f"mean {e.mean():.1f}px  max {e.max():.1f}px", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(args.dir, "check_" + os.path.basename(f)), img)
        per = [float(e.mean()) for e in allv]
        a_ = np.concatenate(allv)
        print(f"{label}: 交点のずれ 平均 {a_.mean():.1f}px  90%点 {np.percentile(a_, 90):.1f}px  "
              f"画像ごとの平均 [{', '.join(f'{v:.0f}' for v in per)}]")
        return a_.mean()

    delta = np.zeros(5)
    e0 = evaluate(R, t, R_bb, t_bb, delta, "\n全部つなげた確認", save=not args.refine)
    if args.cam == "wrist":
        # 診断: 爪で記録した印を信じずに、ボードの位置も画像から合わせたらどうなるか(関節はそのまま)
        cz = None if args.cam_z_cm is None else args.cam_z_cm / 100
        Rd, td, Rbd, tbd, _ = refine(predict, data, R, t, R_bb, t_bb, sb_mm=1e3, sb_deg=1e3, fit_joints=False, quiet=True,
                                     cam_z=cz)
        shift = np.linalg.norm(tbd - t_bb) * 1000
        rot = math.degrees(math.acos(np.clip((np.trace(Rbd.T @ R_bb) - 1) / 2, -1, 1)))
        e_d = evaluate(Rd, td, Rbd, tbd, delta, "診断: ボードの位置も画像から合わせた場合"
                       + ("" if cz is None else f"(カメラの z は {args.cam_z_cm:+.2f}cm に固定)"))
        print(f"  そのときのボードの位置は、爪で記録した印から {shift:.0f}mm、{rot:.1f}度 離れている"
              + ("。※印とボードが合っていない(印を記録したあとにボードが動いた?)。R で消して O・X・Y を記録し直す" if shift > 10 or rot > 3 else ""))
        if e_d < 0.5 * e0 and not args.refine:
            # 手先カメラの取り付けは、ボードの位置を知らなくても(関節値と画像だけで)求められる。こちらのほうがよく合うので採用する
            R, t, R0, t0 = Rd, td, Rd, td
            evaluate(R, t, Rbd, tbd, delta, "  → カメラはこちらの値を使う(印は確認用)", save=True)
            print(f"  カメラの位置(グリッパー基準): [{t[0] * 100:.1f}, {t[1] * 100:.1f}, {t[2] * 100:.1f}] cm")
    if args.refine and args.cam == "wrist":
        t_bb0 = t_bb.copy()
        R, t, R_bb, t_bb, delta = refine(predict, data, R, t, R_bb, t_bb)
        evaluate(R, t, R_bb, t_bb, delta, "関節のずれも一緒に合わせた後", save=True)
        dd = np.degrees(delta)
        print("  関節の原点のずれ [度](シミュの角度に足す): " + "  ".join(f"{j} {v:+.2f}" for j, v in zip(S.JOINTS[:5], dd)))
        board_shift = np.linalg.norm(t_bb - t_bb0) * 1000
        plausible = np.abs(dd[1:4]).max() <= 5 and board_shift <= 10
        if not plausible:
            print("  ※この結果は物理的にありえない(机に触れた確認では、順運動学の誤差は1cm程度、関節では数度以内)。"
                  "関節値と画像の食い違いが、原点のずれでは説明できない(ギアのがた、撮影時の揺れ など)。--apply しない")
        if args.apply and not plausible:
            print("  --apply は取りやめました")
            R, t = R0, t0
        elif args.apply:
            import shutil
            up = os.path.join(HERE, "units.json")
            bak = up + time.strftime(".bak_%Y%m%d_%H%M%S")
            shutil.copy(up, bak)
            units.offset[:5] = units.offset[:5] - units.scale[:5] * delta
            with open(up, "w", encoding="utf-8") as fh:
                json.dump({j: {"scale": round(float(units.scale[i]), 5), "offset": round(float(units.offset[i]), 3)}
                           for i, j in enumerate(S.JOINTS)}, fh, indent=2)
            print(f"  units.json の offset を直しました(元のファイル: {bak})")
        else:
            if plausible:
                print("  units.json と cameras.json をこの結果で直すには --apply を付けて実行し直す。今回は合わせる前の値で保存する")
            R, t = R0, t0
    print("  check_*.png: 緑 = 見つかった交点、赤い輪 = 計算で写るはずの位置")

    if args.cam == "top":  # 右の壁はカメラの所で終わっているので、壁の長さの目安を出す
        mujoco.mj_forward(sim.m, sim.d)
        Rm = sim.d.xmat[sim.m.body("panel_mount").id].reshape(3, 3)
        wall_front = sim.d.geom("wall").xpos - Rm[:, 1] * sim.m.geom_size[sim.m.geom("wall").id][1]
        L = float((wall_front - t) @ Rm[:, 1])
        print(f"  右の壁の長さの目安(パネルの壁からカメラまで): {L * 100:.1f}cm → env_builder.py の PANEL['side_wall']['length']")

    # cameras.json に書く(OpenCV のカメラ: z が視線、y が画像の下 → MuJoCo の lookat / up)
    fwd, down = R[:, 2], R[:, 1]
    cams_path = os.path.join(E.HERE, "cameras.json")
    cams = E.load_json(cams_path, E.CAMERAS)
    h = intr["size"][1]
    cams[args.cam].update(pos=t.round(5).tolist(), lookat=(t + 0.2 * fwd).round(5).tolist(),
                          up=(-down).round(5).tolist(), fovy=round(intr["fovy_deg"], 2), rot180=False,
                          intrinsics=os.path.relpath(os.path.abspath(args.intr), E.HERE))
    for v in cams.values():
        if isinstance(v, dict):
            v.pop("_lens", None)
    with open(cams_path, "w", encoding="utf-8") as fh:
        json.dump(cams, fh, indent=2, ensure_ascii=False)
    print(f"保存: {cams_path}({args.cam} の pos、lookat、up、fovy)")

    # 重ねた画像で確認(シミュは同じ関節値で描く)
    if args.overlay:
        overlay(args, [u[0] for u in used[: args.overlay]])


def refine(predict, data, Rc, tc, Rb, tb, sd_deg=3.0, sb_mm=3.0, sb_deg=2.0, iters=40, fit_joints=True, quiet=False,
           cam_z=None):
    """手先カメラの取り付け、ボードの位置、関節の原点のずれ(5関節)を、全画像の交点のずれが小さくなるように同時に合わせる。
    ボードの位置と関節のずれには「爪で測った値・今の units.json から大きく離れない」弱い制約を付ける。"""
    def unpack(x):
        d = x[:5]
        Rc_ = cv2.Rodrigues(x[5:8])[0] @ Rc
        tc_ = tc + x[8:11]
        Rb_ = cv2.Rodrigues(x[11:14])[0] @ Rb
        tb_ = tb + x[14:17]
        return Rc_, tc_, Rb_, tb_, d

    def resid(x):
        Rc_, tc_, Rb_, tb_, d = unpack(x)
        r = [(predict(Rc_, tc_, Rb_, tb_, d, item) - item[1]).ravel() for item in data]
        r.append(np.degrees(d) / sd_deg)
        r.append(np.degrees(x[11:14]) / sb_deg)
        r.append(x[14:17] * 1000 / sb_mm)
        return np.concatenate(r)

    if cam_z is not None:  # カメラの z(爪の向き)を定規で測った値に固定する
        tc = tc.copy()
        tc[2] = cam_z
    x = np.zeros(17)
    r = resid(x)
    cost, lam = r @ r, 1e-2
    for _ in range(iters):
        J = np.zeros((len(r), 17))
        for k in range(17):
            if (k < 5 and not fit_joints) or (k == 10 and cam_z is not None):  # 固定するものは動かさない
                continue
            dx = np.zeros(17)
            dx[k] = 1e-6
            J[:, k] = (resid(x + dx) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(10):
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            r2 = resid(x + step)
            if r2 @ r2 < cost:
                x, r, cost, lam, improved = x + step, r2, r2 @ r2, lam * 0.3, True
                break
            lam *= 10
        if not improved or np.linalg.norm(step) < 1e-8:
            break
    Rc_, tc_, Rb_, tb_, d = unpack(x)
    ang = math.degrees(np.linalg.norm(x[11:14]))
    if quiet:
        return Rc_, tc_, Rb_, tb_, d
    print(f"  ボードの位置の修正: {np.linalg.norm(x[14:17]) * 1000:.1f}mm、{ang:.1f}度(爪で測った位置から)")
    print(f"  カメラの位置(グリッパー基準): [{tc_[0] * 100:.1f}, {tc_[1] * 100:.1f}, {tc_[2] * 100:.1f}] cm")
    return Rc_, tc_, Rb_, tb_, d


def overlay(args, files):
    import imageio.v2 as imageio
    import so101_sim as S
    sim = S.Sim(args.model_dir)
    units = S.Units(os.path.join(HERE, "units.json"))
    for f in files:
        js = os.path.splitext(f)[0] + ".json"
        if os.path.exists(js):
            sim.reset(units.to_sim(json.load(open(js))["joints_real"]))
        real = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        simg = sim.render()[args.cam]
        out = os.path.join(args.dir, "overlay_" + os.path.basename(f))
        try:
            imageio.imwrite(out, (0.5 * real + 0.5 * simg).astype(np.uint8))
            print(f"重ねた画像: {out}")
        except OSError as e:
            print(f"重ねた画像を保存できませんでした(画像ビューアで開いたままになっていませんか): {out}  {e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board")
    b.add_argument("--out", default="")
    b.add_argument("--board", choices=list(BOARDS), default="fine")
    c = sub.add_parser("capture")
    c.add_argument("--cam", choices=["top", "wrist"], required=True)
    c.add_argument("--index", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--port", default="", help="指定すると関節値も保存し、O・X・Y の印を記録できる")
    c.add_argument("--id", default="my_follower")
    c.add_argument("--board", choices=list(BOARDS), default="fine", help="fine(手先用、15mm)か big(俯瞰用、35mm)")
    c.add_argument("--square", type=float, default=None, help="印刷したマスの実寸 [mm]")
    c.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    g = sub.add_parser("diag")
    g.add_argument("--index", type=int, default=1)
    g.add_argument("--image", default="", help="カメラの代わりに画像ファイルを調べる")
    k = sub.add_parser("check")
    k.add_argument("--cam", choices=["top", "wrist"], default="top")
    k.add_argument("--index", type=int, default=1)
    k.add_argument("--image", default="", help="カメラの代わりに画像ファイルを使う")
    k.add_argument("--port", default="", help="指定すると、今の腕の姿勢をシミュに反映する")
    k.add_argument("--id", default="my_follower")
    k.add_argument("--home", type=float, nargs=6, default=[0.0, -1.83, 1.58, 0.97, 0.0, 0.02])
    k.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    co = sub.add_parser("colors", help="俯瞰の実画像から壁・パネル・液晶・ボタンの色を測って colors.json に書く")
    co.add_argument("--cam", choices=["top", "wrist"], default="top")
    co.add_argument("--index", type=int, default=None, help="カメラの番号(省略時は top=1、wrist=0)")
    co.add_argument("--image", default="")
    co.add_argument("--home", type=float, nargs=6, default=[0.0, -1.83, 1.58, 0.97, 0.0, 0.02])
    co.add_argument("--port", default="", help="指定すると今の腕の姿勢をシミュに反映する(wrist のときに推奨)")
    co.add_argument("--id", default="my_follower")
    co.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    i = sub.add_parser("intrinsics")
    i.add_argument("--dir", required=True)
    i.add_argument("--board", choices=list(BOARDS), default="fine", help="fine(手先用、15mm)か big(俯瞰用、35mm)")
    i.add_argument("--square", type=float, default=None)
    p = sub.add_parser("pose")
    p.add_argument("--cam", choices=["top", "wrist"], required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--intr", required=True)
    p.add_argument("--points", default="", help="board_points.json(省略時は --dir の親フォルダ)")
    p.add_argument("--board", choices=list(BOARDS), default="fine", help="fine(手先用、15mm)か big(俯瞰用、35mm)")
    p.add_argument("--square", type=float, default=None)
    p.add_argument("--overlay", type=int, default=3, help="重ねた画像を何枚作るか")
    p.add_argument("--refine", action="store_true", help="(手先)関節の原点のずれも一緒に合わせる")
    p.add_argument("--cam_z_cm", type=float, default=None,
                   help="(手先)カメラの z [cm](グリッパー基準、爪の向きが -z)を定規で測った値に固定する。"
                        "爪の先端は z=-10.44 なので、z = -10.44 + レンズから爪の先までの距離")
    p.add_argument("--apply", action="store_true", help="--refine の結果で units.json も直す(元のファイルは残す)")
    p.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    args = ap.parse_args()
    {"board": cmd_board, "capture": cmd_capture, "intrinsics": cmd_intrinsics, "pose": cmd_pose, "diag": cmd_diag, "check": cmd_check,
     "colors": cmd_colors}[args.cmd](args)


if __name__ == "__main__":
    main()