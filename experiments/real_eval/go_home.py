"""SO-101 follower をホーム姿勢へ動かす / ホーム姿勢を保存する。

  python go_home.py --port COM4            : 保存済みのホーム姿勢へゆっくり移動(トルクは入れたまま終了)
  python go_home.py --port COM4 --save     : トルクを切る → 手で腕を待機姿勢にする → Enter で保存
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

HERE = Path(__file__).resolve().parent
DEFAULT_POSE = HERE / "home_pose.json"


def make_robot(port: str, robot_id: str) -> SO101Follower:
    # トルクを切らずに切断する(切ると腕が落ちる)。次に lerobot-rollout がこの姿勢を初期姿勢として拾う。
    cfg = SO101FollowerConfig(port=port, id=robot_id, disable_torque_on_disconnect=False)
    return SO101Follower(cfg)


def read_pos(robot: SO101Follower) -> dict[str, float]:
    return {f"{m}.pos": float(v) for m, v in robot.bus.sync_read("Present_Position").items()}


def save(robot: SO101Follower, path: Path) -> None:
    robot.bus.disable_torque()
    input("トルクを切りました。腕を手で待機姿勢(畳んだ姿勢)にして Enter ...")
    pos = read_pos(robot)
    path.write_text(
        json.dumps({"saved_at": datetime.now().isoformat(timespec="seconds"), "pose": pos}, indent=2),
        encoding="utf-8",
    )
    print(f"ホーム姿勢を保存しました: {path}")
    for k, v in pos.items():
        print(f"  {k:20s} {v:8.2f}")
    robot.bus.disconnect(disable_torque=True)


def go(robot: SO101Follower, path: Path, speed: float, tol: float) -> int:
    target = json.loads(path.read_text(encoding="utf-8"))["pose"]
    start = read_pos(robot)
    max_delta = max(abs(target[k] - start[k]) for k in target)
    duration = max(2.0, max_delta / speed)
    print(f"ホーム姿勢へ移動します(最大 {max_delta:.1f}、{duration:.1f} 秒)")

    fps = 50
    steps = max(int(duration * fps), 1)
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)  # なめらかに加減速
        robot.send_action({k: start[k] * (1 - t) + target[k] * t for k in target})
        time.sleep(1 / fps)
    time.sleep(0.8)

    final = read_pos(robot)
    err = {k: final[k] - target[k] for k in target}
    worst = max(err, key=lambda k: abs(err[k]))
    print(f"到達誤差(最大): {worst} {err[worst]:+.2f}")
    # 結果を 1 行の JSON で出す(呼び出し側がログに残す)
    print("HOME_RESULT " + json.dumps({"target": target, "reached": final, "max_abs_err": abs(err[worst])}))
    robot.bus.disconnect(disable_torque=False)
    if abs(err[worst]) > tol:
        print(f"警告: 誤差が {tol} を超えています。腕が何かに当たっていないか確認してください。")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--id", default="my_follower")
    ap.add_argument("--pose", type=Path, default=DEFAULT_POSE)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--speed", type=float, default=30.0, help="関節の最大速度(度/秒 相当)")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    if not a.save and not a.pose.exists():
        print(f"ホーム姿勢が未保存です: {a.pose}  (--save で保存してください)")
        return 1

    robot = make_robot(a.port, a.id)
    robot.connect()
    try:
        if a.save:
            save(robot, a.pose)
            return 0
        return go(robot, a.pose, a.speed, a.tol)
    except BaseException:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
        raise


if __name__ == "__main__":
    sys.exit(main())
