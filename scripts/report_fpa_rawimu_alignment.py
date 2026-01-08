#!/usr/bin/env python3
"""
离线检查 Fixposition RAWIMU(/fixposition/fpa/rawimu + /fixposition/fpa/imubias) 与
机器人 IMU(/imu/data) 以及 FPA ODOMENU(/fixposition/fpa/odomenu) 的坐标系一致性。

输出重点（尽量最小但够用）：
  1) 各 topic 的 frame_id / pose_frame / kin_frame
  2) 静止时重力方向（期望 REP-103/ENU: +Z ≈ +9.8）
  3) /imu/data yaw 与 odomenu yaw 的稳定偏移（只看“是否固定”，不把它当成左右翻转）
  4) 用角速度拟合 RAWIMU(FP_VRTK) -> /imu/data(imu_link) 的最佳旋转矩阵（判断是否存在 180°/轴互换）
  5) (可选) 结合 params.yaml 的 extrinsicRot(IMU->LiDAR) 推出 LiDAR 相对 FP_VRTK 的 yaw

用法：
  source /opt/ros/noetic/setup.bash
  python3 scripts/report_fpa_rawimu_alignment.py --bag ~/autodl-tmp/st_chargeroom_1230_2025-12-30-04-51-18.bag --params config/params.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from itertools import permutations, product
from pathlib import Path
from typing import List, Optional, Tuple


def _import_rosbag():
    try:
        import rosbag  # type: ignore

        return rosbag
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "无法 import rosbag。请先 source ROS 环境，例如：\n"
            "  source /opt/ros/noetic/setup.bash\n"
            "或：\n"
            "  source /root/autodl-tmp/catkin_ws/devel/setup.bash\n"
            f"原始错误: {exc}"
        )


def _import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"无法 import numpy: {exc}")


def _import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"无法 import yaml(PyYAML): {exc}")


def _normalize_frame(frame_id: str) -> str:
    return frame_id.strip().lstrip("/")


def _wrap_pi(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _quat_to_yaw_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_to_R_xyzw(x: float, y: float, z: float, w: float, np):
    # Rotation matrix body->world for ROS quaternion (x,y,z,w)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _circular_mean(angles: List[float]) -> float:
    if not angles:
        return float("nan")
    s = sum(math.sin(a) for a in angles) / len(angles)
    c = sum(math.cos(a) for a in angles) / len(angles)
    return math.atan2(s, c)


def _circular_std(angles: List[float]) -> float:
    if not angles:
        return float("nan")
    s = sum(math.sin(a) for a in angles) / len(angles)
    c = sum(math.cos(a) for a in angles) / len(angles)
    r = math.hypot(c, s)
    r = max(min(r, 1.0), 1e-12)
    return math.sqrt(-2.0 * math.log(r))


def _nearest_index(sorted_times: List[float], t: float) -> Optional[int]:
    if not sorted_times:
        return None
    i = bisect_left(sorted_times, t)
    if i <= 0:
        return 0
    if i >= len(sorted_times):
        return len(sorted_times) - 1
    return i if abs(sorted_times[i] - t) < abs(sorted_times[i - 1] - t) else i - 1


def _yaw_from_R(R, np) -> float:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def _euler_xyz_from_R(R, np) -> Tuple[float, float, float]:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    pitch = math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


def _load_params(params_path: Path, np):
    yaml = _import_yaml()
    data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    lio = (data or {}).get("lio_sam", {})

    def mat(key: str):
        v = lio.get(key, None)
        if not isinstance(v, list) or len(v) != 9:
            return None
        return np.array(v, dtype=float).reshape(3, 3)

    return {
        "lidarFrame": str(lio.get("lidarFrame", "lidar_link")),
        "imuFrame": str(lio.get("imuFrame", "imu_link")),
        "gpsFrame": str(lio.get("gpsFrame", "gps_link")),
        "baseToLidarRot": mat("baseToLidarRot"),
        "gpsExtrinsicRot": mat("gpsExtrinsicRot"),
        "extrinsicRot": mat("extrinsicRot"),
        "extrinsicRPY": mat("extrinsicRPY"),
    }


def main() -> int:
    np = _import_numpy()
    rosbag = _import_rosbag()

    repo_root = Path(__file__).resolve().parents[1]
    default_params = repo_root / "config" / "params.yaml"

    ap = argparse.ArgumentParser(description="Offline RAWIMU alignment report.")
    ap.add_argument("--bag", required=True, help="ROS bag 路径")
    ap.add_argument("--params", default=str(default_params), help="LIO-SAM params.yaml (可选)")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--rawimu-topic", default="/fixposition/fpa/rawimu")
    ap.add_argument("--imubias-topic", default="/fixposition/fpa/imubias")
    ap.add_argument("--odomenu-topic", default="/fixposition/fpa/odomenu")
    ap.add_argument("--acc-samples", type=int, default=3000, help="用于重力统计的样本数")
    ap.add_argument("--max-gyro-pairs", type=int, default=30000, help="用于拟合旋转的 gyro 对齐样本数上限")
    ap.add_argument("--gyro-sync-max-dt", type=float, default=0.003, help="gyro 对齐最大时间差(s)")
    ap.add_argument("--yaw-sync-max-dt", type=float, default=0.02, help="yaw 对齐最大时间差(s)")
    ap.add_argument("--export-json", default="", help="可选：将关键结果导出为 JSON 文件（供脚本自动改 params）")
    args = ap.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    params_path = Path(args.params).expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"bag 不存在: {bag_path}")

    print("=" * 80)
    print("FPA RAWIMU / IMU / ODOMENU 对齐报告 (offline)")
    print("=" * 80)
    print(f"bag: {bag_path}")
    print(f"params: {params_path if params_path.is_file() else '(skip) ' + str(params_path)}")
    print("")

    # Export variables (filled later)
    yaw_mu_rad: Optional[float] = None
    yaw_sd_rad: Optional[float] = None
    yaw_matched: int = 0
    yaw_rate_corr: Optional[float] = None
    yaw_rate_sign_agree: Optional[float] = None

    with rosbag.Bag(str(bag_path), "r") as bag:
        # --------------------------
        # Frames: read first message
        # --------------------------
        def first_msg(topic: str):
            for _tpc, msg, _t in bag.read_messages(topics=[topic]):
                return msg
            return None

        imu0 = first_msg(args.imu_topic)
        raw0 = first_msg(args.rawimu_topic)
        bias0 = first_msg(args.imubias_topic)
        odom0 = first_msg(args.odomenu_topic)

        imu_frame = _normalize_frame(getattr(getattr(imu0, "header", None), "frame_id", "")) if imu0 else "<missing>"
        raw_frame = "<missing>"
        if raw0 is not None and hasattr(raw0, "data"):
            raw_frame = _normalize_frame(getattr(raw0.data.header, "frame_id", ""))
        bias_frame = _normalize_frame(getattr(getattr(bias0, "header", None), "frame_id", "")) if bias0 else "<missing>"

        print("[Topics / Frames]")
        print(f"  {args.imu_topic}: frame_id={imu_frame}")
        print(f"  {args.rawimu_topic}: data.header.frame_id={raw_frame}")
        print(f"  {args.imubias_topic}: frame_id={bias_frame}")
        if odom0 is None:
            print(f"  {args.odomenu_topic}: <missing>")
        else:
            print(
                f"  {args.odomenu_topic}: header.frame_id={_normalize_frame(odom0.header.frame_id)} "
                f"pose_frame={odom0.pose_frame} kin_frame={odom0.kin_frame}"
            )
        print("")

        # --------------------------
        # Load IMU timeline (/imu/data)
        # --------------------------
        imu_t: List[float] = []
        imu_yaw: List[float] = []
        imu_gyr: List[List[float]] = []
        imu_acc: List[List[float]] = []
        for _tpc, msg, t in bag.read_messages(topics=[args.imu_topic]):
            tt = float(t.to_sec())
            imu_t.append(tt)
            imu_yaw.append(
                _quat_to_yaw_xyzw(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
            )
            imu_gyr.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            if len(imu_acc) < int(args.acc_samples):
                imu_acc.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        if not imu_t:
            print("ERROR: /imu/data 为空，无法继续")
            return 2

        imu_gyr_np = np.array(imu_gyr, dtype=float)

        # --------------------------
        # Load IMU bias timeline
        # --------------------------
        bias_t: List[float] = []
        bias_gyr: List[List[float]] = []
        for _tpc, msg, t in bag.read_messages(topics=[args.imubias_topic]):
            bias_t.append(float(t.to_sec()))
            bias_gyr.append([msg.bias_gyr.x, msg.bias_gyr.y, msg.bias_gyr.z])
        bias_t_np = np.array(bias_t, dtype=float) if bias_t else np.zeros((0,), dtype=float)
        bias_gyr_np = np.array(bias_gyr, dtype=float) if bias_gyr else np.zeros((0, 3), dtype=float)

        def bias_at(tt: float):
            if bias_t_np.size == 0:
                return np.zeros(3, dtype=float)
            j = int(np.searchsorted(bias_t_np, tt, side="right") - 1)
            j = max(0, min(j, bias_gyr_np.shape[0] - 1))
            return bias_gyr_np[j]

        # --------------------------
        # Gravity statistics
        # --------------------------
        def acc_stats(name: str, acc_list: List[List[float]]):
            if not acc_list:
                print(f"  {name}: <missing>")
                return
            A = np.array(acc_list, dtype=float)
            mean = A.mean(axis=0)
            std = A.std(axis=0)
            axis = int(np.argmax(np.abs(mean)))
            sign = "+" if mean[axis] >= 0 else "-"
            print(
                f"  {name}: mean=[{mean[0]:+.3f},{mean[1]:+.3f},{mean[2]:+.3f}] "
                f"std=[{std[0]:.3f},{std[1]:.3f},{std[2]:.3f}]  => gravity~{sign}{['X','Y','Z'][axis]}"
            )

        print("[Gravity / Acc]")
        acc_stats("IMU", imu_acc)

        # --------------------------
        # Scan RAWIMU once: collect acc + gyro pairs for fit
        # --------------------------
        raw_t: List[float] = []
        raw_gyr: List[List[float]] = []
        raw_acc: List[List[float]] = []
        pairs_f: List[List[float]] = []
        pairs_i: List[List[float]] = []
        for _tpc, msg, t in bag.read_messages(topics=[args.rawimu_topic]):
            tt = float(t.to_sec())
            m = msg.data
            g = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z], dtype=float) - bias_at(tt)
            raw_t.append(tt)
            raw_gyr.append(g.tolist())
            if len(raw_acc) < int(args.acc_samples):
                raw_acc.append([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])

            idx = _nearest_index(imu_t, tt)
            if idx is None:
                continue
            if abs(imu_t[idx] - tt) > float(args.gyro_sync_max_dt):
                continue
            g_i = imu_gyr_np[idx]
            if float(np.linalg.norm(g)) < 0.1 and float(np.linalg.norm(g_i)) < 0.1:
                continue
            pairs_f.append(g.tolist())
            pairs_i.append(g_i.tolist())
            if len(pairs_f) >= int(args.max_gyro_pairs):
                break

        acc_stats("FPA RAWIMU", raw_acc)
        print("  期望(REP-103/ENU): 静止时加速度应主要在 +Z (~+9.8)")
        print("")

        # --------------------------
        # Yaw offset: odomenu yaw - imu yaw
        # --------------------------
        diffs: List[float] = []
        omega_world_z: List[float] = []
        yaw_rates: List[float] = []
        last_t = None
        last_yaw = None
        for _tpc, msg, t in bag.read_messages(topics=[args.odomenu_topic]):
            tt = float(t.to_sec())
            q = msg.pose.pose.orientation

            if len(diffs) < 5000:
                idx = _nearest_index(imu_t, tt)
                if idx is not None and abs(imu_t[idx] - tt) <= float(args.yaw_sync_max_dt):
                    yaw_odom = _quat_to_yaw_xyzw(q.x, q.y, q.z, q.w)
                    diffs.append(_wrap_pi(yaw_odom - imu_yaw[idx]))

            if raw_t and raw_t[0] <= tt <= raw_t[-1]:
                yaw = _quat_to_yaw_xyzw(q.x, q.y, q.z, q.w)
                if last_t is not None and last_yaw is not None:
                    dt = tt - last_t
                    if 0.0 < dt <= 0.2:
                        yaw_rate = _wrap_pi(yaw - last_yaw) / dt
                        idx_raw = _nearest_index(raw_t, tt)
                        if idx_raw is not None and abs(raw_t[idx_raw] - tt) <= 0.02:
                            Rwb = _quat_to_R_xyzw(q.x, q.y, q.z, q.w, np)
                            omega_b = np.array(raw_gyr[idx_raw], dtype=float)
                            omega_w = Rwb @ omega_b
                            omega_world_z.append(float(omega_w[2]))
                            yaw_rates.append(float(yaw_rate))
                last_t, last_yaw = tt, yaw

            if len(diffs) >= 5000 and (not raw_t or len(yaw_rates) >= 2000):
                break

        print("[Yaw: ODOMENU - /imu/data]")
        if not diffs:
            print("  无法对齐（检查是否存在 /fixposition/fpa/odomenu 与 /imu/data）")
        else:
            mu = _circular_mean(diffs)
            sd = _circular_std(diffs)
            yaw_mu_rad = float(mu)
            yaw_sd_rad = float(sd)
            yaw_matched = int(len(diffs))
            print(f"  matched: {len(diffs)}")
            print(f"  mean diff: {mu * 180.0 / math.pi:+.2f} deg, circ-std: {sd * 180.0 / math.pi:.2f} deg")
            print("  注：稳定偏移通常是世界参考系 yaw 偏置，不等价于“方向相反/镜像”。")
        print("")

    # --------------------------
    # Fit rotation: gyro_fpa -> gyro_imu
    # --------------------------
    print("[Gyro axis alignment: FP_VRTK -> imu_link]")
    if len(pairs_f) < 200:
        print(f"  对齐样本不足: {len(pairs_f)} (<200)，跳过")
        R_i_f = None
    else:
        F = np.array(pairs_f, dtype=float)
        I = np.array(pairs_i, dtype=float)

        F = F - F.mean(axis=0)
        I = I - I.mean(axis=0)

        w = np.linalg.norm(F, axis=1)
        mask = w > 0.2
        F = F[mask]
        I = I[mask]
        w = w[mask]

        if F.shape[0] < 200:
            print(f"  有效样本不足(>0.2rad/s): {int(F.shape[0])}，跳过")
            R_i_f = None
        else:
            H = (I * w[:, None]).T @ F
            U, _S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if float(np.linalg.det(R)) < 0.0:
                U[:, -1] *= -1.0
                R = U @ Vt

            pred = (R @ F.T).T
            rms = float(np.sqrt(np.mean(np.sum((pred - I) ** 2, axis=1))))

            best_rms = None
            best_R = None
            for perm in permutations(range(3)):
                P = np.zeros((3, 3), dtype=float)
                P[range(3), perm] = 1.0
                for signs in product([-1.0, 1.0], repeat=3):
                    M = np.diag(signs) @ P
                    if float(np.linalg.det(M)) < 0.0:
                        continue
                    pred2 = (M @ F.T).T
                    rms2 = float(np.sqrt(np.mean(np.sum((pred2 - I) ** 2, axis=1))))
                    if best_rms is None or rms2 < best_rms:
                        best_rms = rms2
                        best_R = M

            rr, pp, yy = _euler_xyz_from_R(R, np)
            print(f"  matched pairs: {len(pairs_f)} (used {int(F.shape[0])} after gating)")
            print(f"  RMS(residual): {rms:.4f} rad/s")
            print("  R(imu_link <- FP_VRTK):")
            for row in R.tolist():
                print(f"    [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
            print(f"  approx RPY(deg): roll={rr*180/math.pi:+.2f}, pitch={pp*180/math.pi:+.2f}, yaw={yy*180/math.pi:+.2f}")

            if best_R is not None and best_rms is not None:
                Rdiff = R @ best_R.T
                angle = math.acos(max(-1.0, min(1.0, (float(np.trace(Rdiff)) - 1.0) / 2.0))) * 180.0 / math.pi
                print("  nearest signed-permutation (det=+1):")
                for row in best_R.tolist():
                    print(f"    [{row[0]: .0f}, {row[1]: .0f}, {row[2]: .0f}]")
                print(f"    RMS: {best_rms:.4f} rad/s, angle_to_best: {angle:.2f} deg")

            R_i_f = R
    print("")

    # --------------------------
    # Yaw-rate sign sanity: omega_world_z vs yaw_rate (from odomenu)
    # --------------------------
    print("[Yaw-rate sign check: yaw_rate vs omega_world_z]")
    if not raw_t:
        print("  rawimu 为空，跳过")
        print("")
    else:
        if len(yaw_rates) < 50:
            print(f"  样本不足: {len(yaw_rates)}")
            print("")
        else:
            yw = np.array(yaw_rates, dtype=float)
            oz = np.array(omega_world_z, dtype=float)
            mask = np.abs(yw) > 0.05
            if mask.sum() < 50:
                mask = np.abs(yw) > 0.02
            yw = yw[mask]
            oz = oz[mask]
            if yw.size < 50:
                print(f"  转向段样本不足: {int(yw.size)}")
                print("")
            else:
                corr = float(np.corrcoef(yw, oz)[0, 1])
                sign_agree = float((np.sign(yw) == np.sign(oz)).mean())
                yaw_rate_corr = corr
                yaw_rate_sign_agree = sign_agree
                print(f"  samples(turning): {int(yw.size)}")
                print(f"  corr(yaw_rate, omega_world_z): {corr:+.3f}")
                print(f"  sign_agreement: {sign_agree*100:.1f}%")
                if corr < -0.2:
                    print("  ⚠ 很像绕Z方向相反（可能存在 180°/镜像 变换）")
                else:
                    print("  ✓ yaw 正方向与 Z 轴角速度一致（左转 yaw 增大, omega_z 为正）")
                print("")

    # --------------------------
    # Params cross-check (optional)
    # --------------------------
    if params_path.is_file() and R_i_f is not None:
        cfg = _load_params(params_path, np)
        print("[Params cross-check]")
        print(f"  lidarFrame={cfg.get('lidarFrame')} imuFrame={cfg.get('imuFrame')} gpsFrame={cfg.get('gpsFrame')}")
        base_R = cfg.get("baseToLidarRot")
        if base_R is not None:
            print(f"  baseToLidarRot yaw(deg): {_yaw_from_R(base_R, np)*180/math.pi:+.2f}")
        R_l_i_rot = cfg.get("extrinsicRot")
        R_l_i_rpy = cfg.get("extrinsicRPY")

        def report_R(name: str, R):
            if R is None:
                print(f"  {name}: <missing/invalid>")
                return None
            yaw = _yaw_from_R(R, np) * 180.0 / math.pi
            print(f"  {name} (IMU->LiDAR) yaw(deg): {yaw:+.2f}")
            return yaw

        yaw_rot = report_R("extrinsicRot", R_l_i_rot)
        yaw_rpy = report_R("extrinsicRPY", R_l_i_rpy)
        if R_l_i_rot is not None and R_l_i_rpy is not None:
            R_err = R_l_i_rot.T @ R_l_i_rpy
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R_err)) - 1.0) / 2.0))))
            if ang > 0.5:
                print(f"  ⚠ extrinsicRot vs extrinsicRPY mismatch: angle={ang:.2f} deg (LIO-SAM 同时使用两者，建议保持一致)")
            else:
                print(f"  ✓ extrinsicRot == extrinsicRPY (angle={ang:.2f} deg)")

        # Use extrinsicRot (used on acc/gyro) for implied yaw check; fallback to extrinsicRPY if needed.
        R_l_i_used = R_l_i_rot if R_l_i_rot is not None else R_l_i_rpy
        if R_l_i_used is not None:
            R_l_f = R_l_i_used @ R_i_f
            yaw_l_f = _yaw_from_R(R_l_f, np) * 180.0 / math.pi
            print(f"  implied yaw (LiDAR <- FP_VRTK) deg: {yaw_l_f:+.2f}")
            if abs(abs(yaw_l_f) - 180.0) < 15.0:
                print("  ⚠ 接近 180°：很像 lidar 轴反向（或需要在 gpsExtrinsicRot/baseToLidarRot 做 Rz(pi)）")
            elif abs(yaw_l_f) < 15.0:
                print("  ✓ 接近 0°：LiDAR 与 FP 轴基本一致")
        gps_R = cfg.get("gpsExtrinsicRot")
        if gps_R is not None:
            print(f"  gpsExtrinsicRot yaw(deg): {_yaw_from_R(gps_R, np)*180/math.pi:+.2f}")
        print("")

    if str(args.export_json).strip():
        out_path = Path(str(args.export_json)).expanduser().resolve()
        export = {
            "bag": str(bag_path),
            "params": str(params_path) if params_path.is_file() else "",
            "frames": {
                "imu": imu_frame,
                "rawimu": raw_frame,
                "odomenu_parent": _normalize_frame(getattr(odom0.header, "frame_id", "")) if odom0 else "",
                "odomenu_pose_frame": _normalize_frame(getattr(odom0, "pose_frame", "")) if odom0 else "",
            },
            "yaw_diff_odomenu_minus_imu_deg": (yaw_mu_rad * 180.0 / math.pi) if yaw_mu_rad is not None else None,
            "yaw_diff_circ_std_deg": (yaw_sd_rad * 180.0 / math.pi) if yaw_sd_rad is not None else None,
            "yaw_diff_matched": yaw_matched,
            "yaw_rate_corr": yaw_rate_corr,
            "yaw_rate_sign_agreement": yaw_rate_sign_agree,
        }

        # Recommended gpsExtrinsicRot: ENU -> map (yaw-only), derived from yaw(odomenu) - yaw(imu)
        # If ENU yaw is bigger by +d, then map axes are rotated by +d w.r.t ENU, so ENU->map is Rz(-d).
        if yaw_mu_rad is not None:
            yaw_rec_deg = -(yaw_mu_rad * 180.0 / math.pi)
            a = math.radians(yaw_rec_deg)
            c = math.cos(a)
            s = math.sin(a)
            export["recommended_gps_extrinsic_yaw_deg"] = yaw_rec_deg
            export["recommended_gps_extrinsic_rot_rowmajor"] = [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]

        if R_i_f is not None:
            export["gyro_fit_R_imu_link_from_FP_VRTK_rowmajor"] = [float(x) for x in R_i_f.reshape(-1).tolist()]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[export-json] wrote: {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
