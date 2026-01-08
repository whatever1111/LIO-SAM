#!/usr/bin/env python3
"""
运行时验证 TF 关系是否完整（适用于 rosbag 回放 + LIO-SAM 启动后）。

默认检查:
  map -> odom
  odom -> base_link
  base_link -> lidar_link (默认检查是否为 identity，可用 --no-base-lidar-identity 关闭)

用法示例:
  rosrun lio_sam verify_required_tf.py --timeout 20
  python3 scripts/verify_required_tf.py --timeout 20
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List


def _normalize_frame(frame_id: str) -> str:
    return frame_id.strip().lstrip("/")


@dataclass(frozen=True)
class TfCheck:
    parent: str
    child: str
    require_identity: bool = False


def _quat_angle_to_identity(qx: float, qy: float, qz: float, qw: float) -> float:
    # angle = 2*acos(|w|)
    w = max(-1.0, min(1.0, abs(qw)))
    return 2.0 * math.acos(w)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify required TF relations are present (runtime).")
    ap.add_argument("--map", dest="map_frame", default="map", help="map frame")
    ap.add_argument("--odom", dest="odom_frame", default="odom", help="odom frame")
    ap.add_argument("--base", dest="base_frame", default="base_link", help="base_link frame")
    ap.add_argument("--lidar", dest="lidar_frame", default="lidar_link", help="lidar frame")
    ap.add_argument("--timeout", type=float, default=20.0, help="wall-time timeout (seconds)")
    ap.add_argument("--skip-map-odom", action="store_true", help="Skip checking map->odom (dynamic TF)")
    ap.add_argument("--skip-odom-base", action="store_true", help="Skip checking odom->base_link (dynamic TF)")
    ap.add_argument(
        "--no-base-lidar-identity",
        dest="base_lidar_identity",
        action="store_false",
        default=True,
        help="Do not require base->lidar to be identity (useful when lidarFrame is a corrected frame)",
    )
    ap.add_argument("--identity-trans-tol", type=float, default=1e-3, help="identity translation tolerance (m)")
    ap.add_argument("--identity-rot-tol-deg", type=float, default=1e-2, help="identity rotation tolerance (deg)")
    args = ap.parse_args()

    # ROS imports (only when actually used)
    try:
        import rospy  # type: ignore
        import tf2_ros  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "无法 import rospy/tf2_ros。请先 source ROS 环境，例如：\n"
            "  source /opt/ros/noetic/setup.bash\n"
            "或：\n"
            "  source /root/autodl-tmp/catkin_ws/devel/setup.bash\n"
            f"原始错误: {exc}"
        )

    map_frame = _normalize_frame(args.map_frame)
    odom_frame = _normalize_frame(args.odom_frame)
    base_frame = _normalize_frame(args.base_frame)
    lidar_frame = _normalize_frame(args.lidar_frame)

    checks: List[TfCheck] = [
        *([] if args.skip_map_odom else [TfCheck(parent=map_frame, child=odom_frame, require_identity=False)]),
        *([] if args.skip_odom_base else [TfCheck(parent=odom_frame, child=base_frame, require_identity=False)]),
        TfCheck(parent=base_frame, child=lidar_frame, require_identity=bool(args.base_lidar_identity)),
    ]

    rospy.init_node("verify_required_tf", anonymous=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
    _listener = tf2_ros.TransformListener(buf)

    deadline = time.monotonic() + float(args.timeout)
    rot_tol_rad = float(args.identity_rot_tol_deg) * math.pi / 180.0

    last_err = None
    while time.monotonic() < deadline and not rospy.is_shutdown():
        missing = []
        bad_identity = []

        for c in checks:
            try:
                ts = buf.lookup_transform(c.parent, c.child, rospy.Time(0), rospy.Duration(0.2))
            except Exception as exc:
                last_err = exc
                missing.append(f"{c.parent} -> {c.child}")
                continue

            if c.require_identity:
                t = ts.transform.translation
                r = ts.transform.rotation
                t_norm = math.sqrt(t.x * t.x + t.y * t.y + t.z * t.z)
                ang = _quat_angle_to_identity(r.x, r.y, r.z, r.w)
                if t_norm > float(args.identity_trans_tol) or ang > rot_tol_rad:
                    bad_identity.append(
                        f"{c.parent}->{c.child}: |t|={t_norm:.6g}m, angle={ang * 180.0 / math.pi:.6g}deg"
                    )

        if not missing and not bad_identity:
            print("[OK] Required TF tree is available:")
            for c in checks:
                print(f"  - {c.parent} -> {c.child}{' (identity)' if c.require_identity else ''}")
            return 0

        time.sleep(0.2)

    print("[FAIL] TF verification timed out.")
    print("Missing transforms:")
    for s in missing:
        print(f"  - {s}")
    if bad_identity:
        print("Non-identity transforms (expected identity):")
        for s in bad_identity:
            print(f"  - {s}")
    if last_err is not None:
        print(f"Last error: {last_err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
