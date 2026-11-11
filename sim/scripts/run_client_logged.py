"""robot_client を、ステップごとの記録つきで動かすラッパー(LeRobot 0.6.2 / eb7a54c6 のソースに合わせて作成)。

使い方(Windows、PowerShell):
    $env:ACT_LOG = "C:\\Users\\yuya\\lerobot\\logs\\steps_A10_5.csv"
    python run_client_logged.py <robot_client と同じ引数>

CSV の列:
    kind="act"  : 1ステップ実行するごとに1行。a1..a6 = 送った目標関節角、b1..b6 = 送る直前の実際の関節角
    kind="chunk": チャンクを受け取るごとに1行。a1..a6 = チャンク先頭の動作、b1..b6 = チャンク末尾の動作
    timestep    : act は動作の番号、chunk はチャンク先頭の番号
    latest      : その時点で最後に実行した動作の番号
    n           : chunk のときの動作数
"""

import csv
import os
import threading
import time

import lerobot.async_inference.robot_client as rc
from lerobot.utils.import_utils import register_third_party_plugins

_path = os.environ.get("ACT_LOG") or time.strftime("steps_%Y%m%d_%H%M%S.csv")
_f = open(_path, "w", newline="", encoding="utf-8")
_w = csv.writer(_f)
_lock = threading.Lock()
_motors = None  # 例: ["shoulder_pan", ...]


def _header(robot):
    global _motors
    if _motors is None:
        _motors = [k.removesuffix(".pos") for k in robot.action_features]
        _w.writerow(
            ["kind", "t", "timestep", "latest", "n"]
            + [f"a_{m}" for m in _motors]
            + [f"b_{m}" for m in _motors]
        )


def _write(row):
    with _lock:
        _w.writerow(row)
        _f.flush()


_orig_action = rc.RobotClient.control_loop_action
_orig_aggr = rc.RobotClient._aggregate_action_queues


def control_loop_action(self, verbose=False):
    _header(self.robot)
    with self.action_queue_lock:
        nxt = self.action_queue.queue[0] if self.action_queue.qsize() else None
    try:
        pos = self.robot.bus.sync_read("Present_Position")
        b = [pos.get(m, float("nan")) for m in _motors]
    except Exception:  # 通信エラー時は空欄
        b = [float("nan")] * len(_motors)
    out = _orig_action(self, verbose)
    if nxt is not None:
        with self.latest_action_lock:
            latest = self.latest_action
        _write(
            ["act", f"{time.time():.4f}", nxt.get_timestep(), latest, 1]
            + [f"{v:.3f}" for v in nxt.get_action().tolist()]
            + [f"{v:.3f}" for v in b]
        )
    return out


def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
    if incoming_actions:
        _header(self.robot)
        with self.latest_action_lock:
            latest = self.latest_action
        first, last = incoming_actions[0], incoming_actions[-1]
        _write(
            ["chunk", f"{time.time():.4f}", first.get_timestep(), latest, len(incoming_actions)]
            + [f"{v:.3f}" for v in first.get_action().tolist()]
            + [f"{v:.3f}" for v in last.get_action().tolist()]
        )
    return _orig_aggr(self, incoming_actions, aggregate_fn)


rc.RobotClient.control_loop_action = control_loop_action
rc.RobotClient._aggregate_action_queues = _aggregate_action_queues

if __name__ == "__main__":
    print(f"[run_client_logged] 記録先: {os.path.abspath(_path)}")
    register_third_party_plugins()
    try:
        rc.async_client()
    finally:
        _f.close()
