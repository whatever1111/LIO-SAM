#!/usr/bin/env python3
"""
验证 TF 中 “RTK/Fixposition(常见为 FP_*) 坐标系” 与机器人坐标系(base_link 等) 的连通性。

目标：
  - 允许 Fixposition 自己的 TF 子树存在（例如 FP_ECEF->FP_ENU0 / FP_ECEF->FP_POI->FP_VRTK...）
  - 并且要求它与机器人 TF 子树 *连通*（否则无法用 tf 直接对齐/验证外参）

用法：
  python3 scripts/verify_rtk_tf_connection.py --timeout 30 --base base_link
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import rospy
import tf2_ros


def _normalize_frame(s: str) -> str:
    return str(s).strip().lstrip("/")


def _connected(buf: tf2_ros.Buffer, a: str, b: str) -> Tuple[bool, str]:
    """
    Return (connected, detail).
    We consider connected if either a->b or b->a lookup works.
    """
    try:
        buf.lookup_transform(a, b, rospy.Time(0), timeout=rospy.Duration(0.2))
        return True, f"{a} -> {b}"
    except Exception:
        pass
    try:
        buf.lookup_transform(b, a, rospy.Time(0), timeout=rospy.Duration(0.2))
        return True, f"{b} -> {a}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail if RTK(FP_*) frames are NOT connected to the robot TF tree.")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--base", default="base_link", help="robot base frame to test connectivity against")
    ap.add_argument(
        "--rtk-frames",
        nargs="*",
        default=["FP_ENU0", "FP_POI", "FP_VRTK"],
        help="RTK frames that must be connected (default: FP_ENU0 FP_POI FP_VRTK)",
    )
    args = ap.parse_args()

    rospy.init_node("verify_rtk_tf_connection", anonymous=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    _ = tf2_ros.TransformListener(buf)

    base = _normalize_frame(args.base)
    rtk_frames: List[str] = [_normalize_frame(f) for f in args.rtk_frames]

    deadline = time.monotonic() + float(args.timeout)
    last_detail = ""

    while time.monotonic() < deadline and not rospy.is_shutdown():
        not_connected = []
        for f in rtk_frames:
            if f == base:
                continue
            ok, detail = _connected(buf, base, f)
            if not ok:
                not_connected.append((f, detail))

        if not not_connected:
            print("[OK] RTK(FP_*) frames are connected to the robot TF tree:")
            for f in rtk_frames:
                print(f"  - {base} <-> {f}")
            return 0

        last_detail = "; ".join([f"{f}: {d}" for f, d in not_connected])
        time.sleep(0.2)

    print("[FAIL] RTK(FP_*) TF connectivity check timed out.")
    print(f"  base: {base}")
    print(f"  required: {', '.join(rtk_frames)}")
    if last_detail:
        print(f"  last_detail: {last_detail}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
