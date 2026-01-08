#!/usr/bin/env python3
"""
验证 TF 树与 params.yaml(ROS param server) 中外参是否一致。

检查项（仅检查启用的发布项）:
  - lio_sam/publishBaseToLidarTf: base_link -> lidarFrame  vs baseToLidar*
  - lio_sam/publishLidarToImuTf:  lidarFrame -> imuFrame   vs extrinsic*
  - lio_sam/publishBaseToGpsTf:   base_link -> gpsFrame    vs baseToGps*

用法:
  rosrun lio_sam verify_params_tf_consistency.py --timeout 10 --trans-tol 1e-3 --rot-tol-deg 0.5
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rospy
import tf2_ros
from tf.transformations import quaternion_from_matrix


def _normalize_frame(s: str) -> str:
    return str(s).strip().lstrip("/")


def _mat3_from_row_major(v: List[float]) -> np.ndarray:
    if len(v) != 9:
        raise ValueError(f"Expected 9 elements, got {len(v)}")
    return np.asarray(v, dtype=float).reshape(3, 3)


def _vec3(v: List[float]) -> np.ndarray:
    if len(v) != 3:
        raise ValueError(f"Expected 3 elements, got {len(v)}")
    return np.asarray(v, dtype=float).reshape(3)


def _quat_xyzw_from_R(R: np.ndarray) -> Tuple[float, float, float, float]:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    qx, qy, qz, qw = quaternion_from_matrix(T)  # returns (x,y,z,w)
    return float(qx), float(qy), float(qz), float(qw)


def _quat_mul(q1, q2):
    # q = q1 ⊗ q2, xyzw
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_inv(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_angle_deg(q_err) -> float:
    # angle = 2*acos(|w|)
    w = abs(float(q_err[3]))
    w = max(-1.0, min(1.0, w))
    return 2.0 * math.degrees(math.acos(w))


@dataclass(frozen=True)
class ExpectedTf:
    parent: str
    child: str
    t: np.ndarray  # 3,
    q: Tuple[float, float, float, float]  # xyzw


def _read_param(name: str, default=None):
    return rospy.get_param(name, default)


def _build_expected() -> List[ExpectedTf]:
    base = _normalize_frame(_read_param("lio_sam/baselinkFrame", "base_link"))
    lidar = _normalize_frame(_read_param("lio_sam/lidarFrame", "lidar_link"))
    imu = _normalize_frame(_read_param("lio_sam/imuFrame", "imu_link"))
    gps = _normalize_frame(_read_param("lio_sam/gpsFrame", "gps_link"))

    publish_b_l = bool(_read_param("lio_sam/publishBaseToLidarTf", True))
    publish_l_i = bool(_read_param("lio_sam/publishLidarToImuTf", True))
    publish_b_g = bool(_read_param("lio_sam/publishBaseToGpsTf", True))

    expected: List[ExpectedTf] = []

    if publish_b_l:
        t = _vec3(_read_param("lio_sam/baseToLidarTrans", [0.0, 0.0, 0.0]))
        R = _mat3_from_row_major(_read_param("lio_sam/baseToLidarRot", [1, 0, 0, 0, 1, 0, 0, 0, 1]))
        expected.append(ExpectedTf(base, lidar, t, _quat_xyzw_from_R(R)))

    if publish_l_i:
        t_raw = _vec3(_read_param("lio_sam/extrinsicTrans", [0.0, 0.0, 0.0]))
        trans_is_lidar_to_imu = bool(_read_param("lio_sam/extrinsicTransIsLidarToImu", True))
        # Prefer extrinsicRPY for rotation; fallback to extrinsicRot.
        rot_list = _read_param("lio_sam/extrinsicRPY", None)
        if not isinstance(rot_list, list) or len(rot_list) != 9:
            rot_list = _read_param("lio_sam/extrinsicRot", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        R_l_i = _mat3_from_row_major(rot_list)

        t = t_raw
        if not trans_is_lidar_to_imu:
            # Given t_i_l, convert to t_l_i = -R_l_i * t_i_l
            t = -(R_l_i @ t_raw)

        expected.append(ExpectedTf(lidar, imu, t, _quat_xyzw_from_R(R_l_i)))

    if publish_b_g:
        t = _vec3(_read_param("lio_sam/baseToGpsTrans", [0.0, 0.0, 0.0]))
        R = _mat3_from_row_major(_read_param("lio_sam/baseToGpsRot", [1, 0, 0, 0, 1, 0, 0, 0, 1]))
        expected.append(ExpectedTf(base, gps, t, _quat_xyzw_from_R(R)))

    return expected


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify TF vs lio_sam params.")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--trans-tol", type=float, default=1e-3, help="m")
    ap.add_argument("--rot-tol-deg", type=float, default=0.5, help="deg")
    args = ap.parse_args()

    rospy.init_node("verify_params_tf_consistency", anonymous=True)
    buf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
    _ = tf2_ros.TransformListener(buf)

    expected = _build_expected()
    if not expected:
        print("No TF checks enabled by params (all publish* flags are false).")
        return 0

    ok = True
    for e in expected:
        try:
            tr = buf.lookup_transform(e.parent, e.child, rospy.Time(0), timeout=rospy.Duration(args.timeout))
        except Exception as exc:
            print(f"[FAIL] missing TF {e.parent} -> {e.child}: {exc}")
            ok = False
            continue

        t_act = np.array(
            [
                tr.transform.translation.x,
                tr.transform.translation.y,
                tr.transform.translation.z,
            ],
            dtype=float,
        )
        q_act = (
            float(tr.transform.rotation.x),
            float(tr.transform.rotation.y),
            float(tr.transform.rotation.z),
            float(tr.transform.rotation.w),
        )

        terr = float(np.linalg.norm(t_act - e.t))
        q_err = _quat_mul(_quat_inv(e.q), q_act)
        rerr = _quat_angle_deg(q_err)

        pass_t = terr <= float(args.trans_tol)
        pass_r = rerr <= float(args.rot_tol_deg)
        status = "OK" if (pass_t and pass_r) else "FAIL"
        print(
            f"[{status}] {e.parent}->{e.child}: "
            f"trans_err={terr:.6f} m (tol {args.trans_tol}), rot_err={rerr:.3f} deg (tol {args.rot_tol_deg})"
        )
        if not (pass_t and pass_r):
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

