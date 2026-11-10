"""シミュのカメラ(top / wrist)を、実機の画像に重ねながらスライダーで合わせる。

使い方(画面のある PC で):
    python tune_camera.py --cam top --ref ref_top.png
    python tune_camera.py --cam wrist --ref ref_wrist.png
    python tune_camera.py --cam top --ref episode_000000.mp4 --frame 0     # 動画の1コマを使う
    python tune_camera.py --cam top --capture 1                          # 実機のカメラ(index 1)から1枚撮る
キー: s = cameras.json に保存 / r = 元に戻す / q または Esc = 終了
参照画像を撮ったときの腕の姿勢を --q_real(実機の値、units.json で変換)か --home [rad] で渡すと、腕の重なりも比べられる。
top の位置と見る点はロボットの土台基準 [cm]、wrist は gripper ボディ基準 [mm](指先の向きが -z)。
"""
import argparse
import json
import os

import cv2
import mujoco
import numpy as np

import env_builder as E

# スライダー: (名前, 設定のキー, 添字, 最小, 最大)。値は top は cm、wrist は mm
SLIDERS = [("pos x", "pos", 0), ("pos y", "pos", 1), ("pos z", "pos", 2),
           ("look x", "lookat", 0), ("look y", "lookat", 1), ("look z", "lookat", 2)]
RANGE = {"top": (-100, 150, 100.0), "wrist": (-200, 200, 1000.0)}  # (最小, 最大, m からの倍率)


def load_ref(args, w, h):
    if args.capture is not None:
        cap = cv2.VideoCapture(args.capture)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(10):  # 露出が落ち着くまで数枚捨てる
            ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"カメラ {args.capture} から撮れませんでした")
        cv2.imwrite(f"ref_{args.cam}.png", img)
        print(f"撮った画像を保存しました: ref_{args.cam}.png")
    elif args.ref.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(args.ref)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"動画を読めませんでした(AV1 の動画は OpenCV で読めないことがある): {args.ref}")
    else:
        img = cv2.imread(args.ref)
        if img is None:
            raise SystemExit(f"画像を読めませんでした: {args.ref}")
    return cv2.resize(img, (w, h))  # BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.expanduser("~/SO-ARM100/Simulation/SO101"))
    ap.add_argument("--cam", choices=["top", "wrist"], default="top")
    ap.add_argument("--ref", default="")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--capture", type=int, default=None, help="実機のカメラの index")
    ap.add_argument("--home", type=float, nargs=6, default=[0.0, -1.83, 1.58, 0.97, 0.0, 0.02])
    ap.add_argument("--q_real", type=float, nargs=6, help="参照画像のときの関節(実機の値)。units.json で変換する")
    ap.add_argument("--panel_pos", type=float, nargs=3)
    ap.add_argument("--panel_rpy", type=float, nargs=3)
    args = ap.parse_args()
    if not args.ref and args.capture is None:
        raise SystemExit("--ref か --capture を指定してください")

    cams_path = os.path.join(E.HERE, "cameras.json")
    cams = E.load_json(cams_path, E.CAMERAS)
    model, _ = E.load_environment(args.model_dir, args.panel_pos, args.panel_rpy, cameras=cams)
    data = mujoco.MjData(model)
    q = np.array(args.home)
    if args.q_real:
        import so101_sim
        q = so101_sim.Units(os.path.join(E.HERE, "units.json")).to_sim(args.q_real)
    data.qpos[[model.jnt_qposadr[model.joint(j).id] for j in E.JOINTS]] = q
    mujoco.mj_forward(model, data)

    w, h = cams["img_w"], cams["img_h"]
    ref = load_ref(args, w, h)
    renderer = mujoco.Renderer(model, h, w)
    cid = model.camera(args.cam).id
    lo, hi, k = RANGE[args.cam]
    c0 = json.loads(json.dumps(cams[args.cam]))  # 元の値(r で戻す)

    win = f"tune {args.cam}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def set_sliders(c):
        for name, key, i in SLIDERS:
            cv2.setTrackbarPos(name, win, int(round(c[key][i] * k)) - lo)
        cv2.setTrackbarPos("fovy", win, int(round(c["fovy"])))
        cv2.setTrackbarPos("rot180", win, int(bool(c.get("rot180"))))

    for name, key, i in SLIDERS:
        cv2.createTrackbar(name, win, 0, hi - lo, lambda v: None)
    cv2.createTrackbar("fovy", win, 50, 120, lambda v: None)
    cv2.createTrackbar("rot180", win, 0, 1, lambda v: None)
    cv2.createTrackbar("blend %", win, 50, 100, lambda v: None)
    set_sliders(cams[args.cam])

    unit = "cm" if args.cam == "top" else "mm"
    print(f"位置と見る点のスライダーは、目盛りから {-lo} を引いた値が {unit}(画面下に実際の値を表示)。s で保存、r で元に戻す、q で終了")
    while True:
        c = cams[args.cam]
        for name, key, i in SLIDERS:
            c[key][i] = (cv2.getTrackbarPos(name, win) + lo) / k
        c["fovy"] = max(5, cv2.getTrackbarPos("fovy", win))
        c["rot180"] = bool(cv2.getTrackbarPos("rot180", win))
        if np.linalg.norm(np.subtract(c["lookat"], c["pos"])) > 1e-6:
            model.cam_pos[cid] = c["pos"]
            model.cam_quat[cid] = E.cam_quat(c["pos"], c["lookat"], c["up"])
        model.cam_fovy[cid] = c["fovy"]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=args.cam)
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        if c["rot180"]:
            sim = sim[::-1, ::-1]
        a = cv2.getTrackbarPos("blend %", win) / 100.0
        view = cv2.addWeighted(np.ascontiguousarray(sim), a, ref, 1 - a, 0)
        txt = f"pos {np.round(np.array(c['pos']) * k, 1)} look {np.round(np.array(c['lookat']) * k, 1)} {unit}  fovy {c['fovy']}"
        cv2.putText(view, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            with open(cams_path, "w", encoding="utf-8") as f:
                json.dump(cams, f, indent=2, ensure_ascii=False)
            cv2.imwrite(f"tuned_{args.cam}.png", view)
            print(f"保存しました: {cams_path}(重ねた画像は tuned_{args.cam}.png)")
        if key == ord("r"):
            cams[args.cam] = json.loads(json.dumps(c0))
            set_sliders(cams[args.cam])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
