#!/usr/bin/env python3
"""
Verify the derived yaw (~45 deg) between Fixposition RTK frames and LiDAR using the same rotation chain
used in the offline analysis:

  LiDAR <- FP_POI  = (LiDAR <- IMU) * (IMU <- FP_VRTK) * (FP_VRTK <- FP_POI)

Inputs:
  - params.yaml: provides extrinsicRot/extrinsicRPY (LiDAR <- IMU) and baseToLidarRot
  - bag /tf_static: provides FP_POI -> FP_VRTK (static)
  - alignment json: provides gyro-fit rotation IMU <- FP_VRTK
      (exported by scripts/report_fpa_rawimu_alignment.py --export-json ...)

This script prints each component yaw and the composed yaw.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional, Tuple

import numpy as np
import rosbag
import yaml


def _normalize_frame(s: str) -> str:
    return str(s or "").strip().lstrip("/")


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, 2] *= -1.0
        Rn = U @ Vt
    return Rn


def _quat_to_R_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Returns R such that v_parent = R * v_child
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return _project_to_so3(R)


def _rpy_deg_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    # ZYX (roll-pitch-yaw) consistent with ROS tf::Matrix3x3.getRPY
    R = np.asarray(R, dtype=float).reshape(3, 3)
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    pitch = math.asin(-float(R[2, 0]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _yaw_deg_from_R(R: np.ndarray) -> float:
    return _rpy_deg_from_R(R)[2]


def _load_mat3_row_major(v) -> Optional[np.ndarray]:
    if not isinstance(v, list) or len(v) != 9:
        return None
    return np.array([float(x) for x in v], dtype=float).reshape(3, 3)


def _load_params(params_file: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(params_file, "r") as f:
        data = yaml.safe_load(f) or {}
    lio = data.get("lio_sam") or {}

    base_rot = _load_mat3_row_major(lio.get("baseToLidarRot", None))
    if base_rot is None:
        base_rot = np.eye(3, dtype=float)

    # Prefer extrinsicRPY if present (matches LIO-SAM usage); fall back to extrinsicRot.
    ext = _load_mat3_row_major(lio.get("extrinsicRPY", None))
    if ext is None:
        ext = _load_mat3_row_major(lio.get("extrinsicRot", None))
    if ext is None:
        ext = np.eye(3, dtype=float)
    return _project_to_so3(base_rot), _project_to_so3(ext)


def _extract_static_tf_R(bag_path: str, parent: str, child: str) -> Optional[np.ndarray]:
    parent_n = _normalize_frame(parent)
    child_n = _normalize_frame(child)
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, _t in bag.read_messages(topics=["/tf_static"]):
            for tr in msg.transforms:
                if _normalize_frame(tr.header.frame_id) == parent_n and _normalize_frame(tr.child_frame_id) == child_n:
                    q = tr.transform.rotation
                    return _quat_to_R_xyzw(q.x, q.y, q.z, q.w)
    return None


def _load_alignment_R(json_path: str) -> Optional[np.ndarray]:
    with open(json_path, "r") as f:
        data = json.load(f) or {}
    v = data.get("gyro_fit_R_imu_link_from_FP_VRTK_rowmajor", None)
    if not isinstance(v, list) or len(v) != 9:
        return None
    return _project_to_so3(np.array([float(x) for x in v], dtype=float).reshape(3, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the derived ~45 deg RTK(FP_POI) <-> LiDAR yaw chain.")
    ap.add_argument("--bag", required=True, help="Input bag (must contain /tf_static FP_POI->FP_VRTK)")
    ap.add_argument("--params", required=True, help="LIO-SAM params.yaml (extrinsicRot/extrinsicRPY, baseToLidarRot)")
    ap.add_argument(
        "--assume-lidar-imu-aligned",
        action="store_true",
        help="Assume lidar_link and imu_link axes are aligned: set R_lidar_imu = Identity (ignore extrinsicRPY/Rot in params).",
    )
    ap.add_argument(
        "--alignment-json",
        default="/tmp/fpa_rawimu_alignment.json",
        help="JSON exported by report_fpa_rawimu_alignment.py (default: /tmp/fpa_rawimu_alignment.json)",
    )
    ap.add_argument("--poi-frame", default="FP_POI")
    ap.add_argument("--vrtk-frame", default="FP_VRTK")
    ap.add_argument("--imu-frame", default="imu_link")
    ap.add_argument("--lidar-frame", default="lidar_link")
    args = ap.parse_args()

    bag_path = os.path.expanduser(args.bag)
    params_path = os.path.expanduser(args.params)
    align_path = os.path.expanduser(args.alignment_json)

    if not os.path.isfile(bag_path):
        raise SystemExit(f"bag not found: {bag_path}")
    if not os.path.isfile(params_path):
        raise SystemExit(f"params not found: {params_path}")
    if not os.path.isfile(align_path):
        raise SystemExit(f"alignment json not found: {align_path} (run report_fpa_rawimu_alignment.py --export-json first)")

    R_b_l, R_l_i = _load_params(params_path)
    if args.assume_lidar_imu_aligned:
        R_l_i = np.eye(3, dtype=float)
    R_i_v = _load_alignment_R(align_path)
    if R_i_v is None:
        raise SystemExit("alignment json missing gyro_fit_R_imu_link_from_FP_VRTK_rowmajor")

    R_p_v = _extract_static_tf_R(bag_path, args.poi_frame, args.vrtk_frame)
    if R_p_v is None:
        raise SystemExit(f"missing /tf_static transform: {args.poi_frame} -> {args.vrtk_frame}")
    R_v_p = R_p_v.T

    # Compose
    R_l_v = R_l_i @ R_i_v
    R_l_p = R_l_v @ R_v_p

    # If base and lidar are different, also report base<-POI
    R_b_p = R_b_l @ R_l_p

    print("==== Inputs ====")
    print(f"bag: {bag_path}")
    print(f"params: {params_path}")
    print(f"alignment-json: {align_path}")
    print("")

    print("==== Component rotations (R_A_B: A <- B) ====")
    r, p, y = _rpy_deg_from_R(R_b_l)
    print(f"R_base_lidar   (base <- lidar): rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")

    r, p, y = _rpy_deg_from_R(R_l_i)
    print(f"R_lidar_imu    (lidar <- imu):  rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")

    r, p, y = _rpy_deg_from_R(R_i_v)
    print(f"R_imu_vrtk     (imu <- vrtk):   rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")

    r, p, y = _rpy_deg_from_R(R_p_v)
    print(f"R_poi_vrtk     (poi <- vrtk):   rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")
    r, p, y = _rpy_deg_from_R(R_v_p)
    print(f"R_vrtk_poi     (vrtk <- poi):   rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")

    print("")
    print("==== Composed rotations ====")
    r, p, y = _rpy_deg_from_R(R_l_v)
    print(f"R_lidar_vrtk   (lidar <- vrtk): rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")
    r, p, y = _rpy_deg_from_R(R_l_p)
    print(f"R_lidar_poi    (lidar <- poi):  rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")
    r, p, y = _rpy_deg_from_R(R_b_p)
    print(f"R_base_poi     (base <- poi):   rpy_deg=[{r:+.2f},{p:+.2f},{y:+.2f}]")

    print("")
    print("==== Summary ====")
    print(f"yaw(lidar <- poi) = {_yaw_deg_from_R(R_l_p):+.2f} deg  (expected ~+45 deg)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
