#!/usr/bin/env python3
"""
验证 Fixposition ENU 坐标系与 LIO-SAM map 坐标系的桥接 TF 是否与 params.yaml 中 gpsExtrinsicRot 一致。

约定：
  - LIO-SAM 使用 gpsExtrinsicRot (R_map_enu) 把 GPS/ENU 坐标旋到 map/lidar 坐标：
      p_map = R_map_enu * p_enu
  - TF bridge 发布的是 parent=FP_ENU0, child=map：
      p_enu = R_enu_map * p_map + t
    因此应满足：R_enu_map = (R_map_enu)^T

用法：
  python3 scripts/verify_fpa_map_tf_consistency.py --timeout 60 --enu FP_ENU0 --map map
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List

import numpy as np
import rospy
import tf2_ros


def _normalize_frame(s: str) -> str:
    return str(s).strip().lstrip("/")


def _mat3_from_row_major(v: List[float]) -> np.ndarray:
    if not isinstance(v, list) or len(v) != 9:
        raise ValueError("gpsExtrinsicRot must be a 9-element list")
    return np.asarray([float(x) for x in v], dtype=float).reshape(3, 3)


def _R_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # ROS quaternion order: (x,y,z,w)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    v = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(v))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify FP_ENU0->map TF matches gpsExtrinsicRot.")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--enu", default="FP_ENU0")
    ap.add_argument("--map", dest="map_frame", default="map")
    ap.add_argument("--rot-tol-deg", type=float, default=0.5)
    args = ap.parse_args()

    rospy.init_node("verify_fpa_map_tf_consistency", anonymous=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    _ = tf2_ros.TransformListener(buf)

    enu = _normalize_frame(args.enu)
    mapf = _normalize_frame(args.map_frame)

    try:
        R_map_enu = _mat3_from_row_major(rospy.get_param("lio_sam/gpsExtrinsicRot"))
    except Exception as exc:
        print(f"[FAIL] cannot read lio_sam/gpsExtrinsicRot: {exc}")
        return 2
    R_enu_map_expected = R_map_enu.T

    deadline = time.monotonic() + float(args.timeout)
    last_err = None
    while time.monotonic() < deadline and not rospy.is_shutdown():
        try:
            tr = buf.lookup_transform(enu, mapf, rospy.Time(0), timeout=rospy.Duration(0.5))
        except Exception as exc:
            last_err = exc
            time.sleep(0.1)
            continue

        q = tr.transform.rotation
        R_enu_map_actual = _R_from_quat_xyzw(float(q.x), float(q.y), float(q.z), float(q.w))
        R_err = R_enu_map_expected.T @ R_enu_map_actual  # expected^-1 * actual, but expected is orthonormal
        ang = _angle_deg(R_err)

        if ang <= float(args.rot_tol_deg):
            print(f"[OK] {enu}->{mapf} rotation matches gpsExtrinsicRot (err={ang:.3f} deg)")
            return 0

        print(f"[FAIL] {enu}->{mapf} rotation mismatch: err={ang:.3f} deg (tol {args.rot_tol_deg})")
        return 2

    print("[FAIL] timed out waiting for TF or params")
    if last_err is not None:
        print(f"  last_err: {last_err}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

