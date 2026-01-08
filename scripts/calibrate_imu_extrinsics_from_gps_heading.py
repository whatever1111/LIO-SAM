#!/usr/bin/env python3
"""
基于 /imu/data 与 Fixposition FPA(odomenu) 的航向，离线标定 LIO-SAM IMU 外参(绕Z)。

输入:
  - IMU:  /imu/data (sensor_msgs/Imu)
  - GPS:  /fixposition/fpa/odomenu (fixposition_driver_msgs/FpaOdomenu)
  - (可选) LiDAR: /lidar_points (sensor_msgs/PointCloud2) 用于自动判断 LiDAR 前向轴 (+X/-X/+Y/-Y)

输出:
  - 误差曲线(随时间)、直方图、滑窗外参漂移曲线
  - CSV(逐样本) 与 summary.json(统计)
  - 推荐的 extrinsicRot / extrinsicRPY (默认在当前 params.yaml 的基础上仅做绕Z微调)

注意:
  - 本bag存在 header.stamp 与 rosbag 记录时间(t)不一致的情况(例如 FPA 消息 header 使用 GNSS 时标)。
    本脚本默认用 rosbag 的 t.to_sec() 作为对齐时间基准，可用 --time-source header 切换。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rosbag  # noqa: E402
import yaml  # noqa: E402


def _wrap_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _circular_mean(angles: np.ndarray) -> float:
    if angles.size == 0:
        return float("nan")
    return math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))


def _circular_std(angles: np.ndarray) -> float:
    if angles.size == 0:
        return float("nan")
    r = math.hypot(float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles))))
    r = max(min(r, 1.0), 1e-12)
    return math.sqrt(-2.0 * math.log(r))


def _quat_to_yaw_rad_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_xyzw_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _yaw_from_rotmat(R_w_b: np.ndarray) -> float:
    # ZYX yaw (same as quaternion yaw formula above)
    return math.atan2(float(R_w_b[1, 0]), float(R_w_b[0, 0]))


def _rotz(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _format_rot_row_major(mat: np.ndarray, decimals: int = 6) -> List[float]:
    mat = np.asarray(mat, dtype=float).reshape(3, 3)
    return [round(float(v), decimals) for v in mat.reshape(-1).tolist()]


def _matrix_from_row_major(values: List[float]) -> np.ndarray:
    if len(values) != 9:
        raise ValueError(f"Expected 9 elements, got {len(values)}")
    return np.array(values, dtype=float).reshape(3, 3)


def _infer_lidar_forward_axis(
    bag_path: str,
    lidar_topic: str,
    max_frames: int = 3,
    max_points_per_frame: int = 20000,
) -> Tuple[Optional[str], Dict[str, float]]:
    """
    粗略推断 LiDAR “前方”对应哪个轴。
    对于前向FOV的雷达，一般会出现某个轴(±X/±Y)占多数。
    """

    x_pos = x_neg = y_pos = y_neg = 0
    total = 0
    frames = 0

    try:
        bag = rosbag.Bag(bag_path, "r")
    except Exception:
        return None, {}

    with bag:
        for _, msg, _t in bag.read_messages(topics=[lidar_topic]):
            offsets = {f.name: f.offset for f in msg.fields}
            if "x" not in offsets or "y" not in offsets:
                break

            data = msg.data
            step = msg.point_step
            n = min(int(msg.width * msg.height), max_points_per_frame)
            for i in range(n):
                base = i * step
                x = float(np.frombuffer(data, dtype=np.float32, count=1, offset=base + offsets["x"])[0])
                y = float(np.frombuffer(data, dtype=np.float32, count=1, offset=base + offsets["y"])[0])
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                total += 1
                if x > 0:
                    x_pos += 1
                elif x < 0:
                    x_neg += 1
                if y > 0:
                    y_pos += 1
                elif y < 0:
                    y_neg += 1

            frames += 1
            if frames >= max_frames:
                break

    if total <= 0:
        return None, {}

    ratios = {
        "+X": x_pos / total,
        "-X": x_neg / total,
        "+Y": y_pos / total,
        "-Y": y_neg / total,
    }

    # 简单判别：若某个方向占比足够高，则认为是“前向”
    best_axis, best_ratio = max(ratios.items(), key=lambda kv: kv[1])
    if best_ratio >= 0.6:
        return best_axis, ratios
    return None, ratios


def _lidar_forward_axis_to_target_offset(forward_axis: str) -> float:
    """
    将“前向轴”映射到 LiDAR x轴 与 前向夹角(目标航向偏移)：
      - forward=+X: x轴即前向 => offset=0
      - forward=-X: x轴为后向 => offset=pi
      - forward=+Y: 假设 y前/x右 => x 比前向顺时针90deg => offset=-pi/2
      - forward=-Y: 假设 -y前/x左 => x 比前向逆时针90deg => offset=+pi/2
    """

    if forward_axis == "+X":
        return 0.0
    if forward_axis == "-X":
        return math.pi
    if forward_axis == "+Y":
        return -math.pi / 2.0
    if forward_axis == "-Y":
        return math.pi / 2.0
    raise ValueError(f"Unsupported forward axis: {forward_axis}")


def _read_time(t_bag, msg, time_source: str) -> float:
    if time_source == "bag":
        return float(t_bag.to_sec())
    if time_source == "header":
        if not hasattr(msg, "header"):
            raise ValueError("Message has no header, cannot use header time")
        return float(msg.header.stamp.to_sec())
    raise ValueError(f"Unknown time source: {time_source}")


def _load_params_extrinsics(params_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(params_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "lio_sam" not in data:
        raise ValueError("params.yaml missing 'lio_sam' root key")
    cfg = data["lio_sam"]
    ext_rot = _matrix_from_row_major(cfg["extrinsicRot"])
    ext_rpy = _matrix_from_row_major(cfg["extrinsicRPY"])
    return ext_rot, ext_rpy


def _patch_params_yaml_inplace(params_path: str, new_ext_rot: np.ndarray, new_ext_rpy: np.ndarray) -> None:
    """
    保留注释，按块替换 extrinsicRot/extrinsicRPY 的 [] 列表。
    """

    def replace_block(lines: List[str], key: str, values: List[float]) -> List[str]:
        start_idx = None
        indent = ""
        for i, line in enumerate(lines):
            if line.lstrip().startswith(f"{key}:"):
                start_idx = i
                indent = line[: len(line) - len(line.lstrip())]
                break
        if start_idx is None:
            raise ValueError(f"Key '{key}' not found in {params_path}")

        end_idx = start_idx
        # consume until closing bracket
        while end_idx < len(lines) and "]" not in lines[end_idx]:
            end_idx += 1
        if end_idx >= len(lines):
            raise ValueError(f"Unterminated list for '{key}' in {params_path}")

        # format as 3 rows
        v = values
        row0 = f"{indent}{key}: [{v[0]}, {v[1]}, {v[2]},\n"
        row1 = f"{indent}{' ' * (len(key) + 2)}{v[3]}, {v[4]}, {v[5]},\n"
        row2 = f"{indent}{' ' * (len(key) + 2)}{v[6]}, {v[7]}, {v[8]}]\n"
        new_block = [row0, row1, row2]

        return lines[:start_idx] + new_block + lines[end_idx + 1 :]

    with open(params_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rot_values = _format_rot_row_major(new_ext_rot, decimals=6)
    rpy_values = _format_rot_row_major(new_ext_rpy, decimals=6)

    lines = replace_block(lines, "extrinsicRot", rot_values)
    lines = replace_block(lines, "extrinsicRPY", rpy_values)

    with open(params_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


@dataclasses.dataclass
class Sample:
    t: float
    t_rel: float
    speed: float
    yaw_target: float
    yaw_lidar_current: float
    err_current: float
    yaw_lidar_calib: float
    err_calib: float


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate LIO-SAM IMU extrinsics (yaw) using Fixposition FPA odomenu heading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bag", required=True, help="rosbag 路径")
    parser.add_argument("--params", default="config/params.yaml", help="LIO-SAM params.yaml 路径")
    parser.add_argument("--imu-topic", default="/imu/data", help="IMU话题")
    parser.add_argument("--gps-topic", default="/fixposition/fpa/odomenu", help="FPA odomenu话题")
    parser.add_argument("--lidar-topic", default="/lidar_points", help="LiDAR话题(用于判断前向轴)")
    parser.add_argument(
        "--gps-heading-source",
        choices=["quat", "pos"],
        default="quat",
        help="GPS航向来源：quat=使用odomenu姿态yaw；pos=使用ENU位置差分航向",
    )
    parser.add_argument(
        "--time-source",
        choices=["bag", "header"],
        default="bag",
        help="对齐时间基准：bag=使用rosbag记录时间t；header=使用msg.header.stamp",
    )
    parser.add_argument("--sync-tol", type=float, default=0.05, help="IMU与GPS最近邻匹配容差(s)")
    parser.add_argument("--min-speed", type=float, default=0.5, help="最小速度阈值(m/s)，用于筛选可靠航向")
    parser.add_argument("--max-samples", type=int, default=0, help="最多使用多少个有效样本(0=不限)")
    parser.add_argument(
        "--lidar-forward",
        choices=["auto", "+X", "-X", "+Y", "-Y"],
        default="auto",
        help="LiDAR前向轴，auto会从点云分布估计",
    )
    parser.add_argument("--out-dir", default="output/imu_gps_extrinsic_calib", help="输出目录")
    parser.add_argument("--window-sec", type=float, default=10.0, help="滑窗估计外参的窗口长度(s)")
    parser.add_argument("--write-params", action="store_true", help="将推荐外参写回 params.yaml (保留注释)")

    args = parser.parse_args(argv)

    bag_path = os.path.expanduser(args.bag)
    params_path = os.path.expanduser(args.params)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext_rot_old, ext_rpy_old = _load_params_extrinsics(params_path)
    if not np.allclose(ext_rot_old, ext_rpy_old, atol=1e-6):
        print(
            "[WARN] params.yaml 中 extrinsicRot 与 extrinsicRPY 不一致；本脚本会对两者都施加同一绕Z微调。",
            file=sys.stderr,
        )

    # 估计LiDAR前向轴
    forward_axis = None
    forward_ratios: Dict[str, float] = {}
    if args.lidar_forward == "auto":
        forward_axis, forward_ratios = _infer_lidar_forward_axis(bag_path, args.lidar_topic)
        if forward_axis is None:
            forward_axis = "-X"  # 保守默认：Livox 常见为 -X 前
            print(
                f"[WARN] 无法可靠估计 LiDAR 前向轴，默认使用 {forward_axis}；你也可以用 --lidar-forward 指定。",
                file=sys.stderr,
            )
        else:
            print(f"[INFO] LiDAR前向轴估计: {forward_axis} (ratios={forward_ratios})")
    else:
        forward_axis = args.lidar_forward

    target_offset = _lidar_forward_axis_to_target_offset(forward_axis)

    # 读取 IMU: 用 bag时间对齐
    imu_t: List[float] = []
    imu_q: List[Tuple[float, float, float, float]] = []  # x,y,z,w
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, t in bag.read_messages(topics=[args.imu_topic]):
            ts = _read_time(t, msg, args.time_source)
            q = msg.orientation
            imu_t.append(ts)
            imu_q.append((float(q.x), float(q.y), float(q.z), float(q.w)))

    if len(imu_t) < 10:
        print(f"[ERROR] IMU样本过少: {len(imu_t)}", file=sys.stderr)
        return 2

    imu_t_np = np.asarray(imu_t, dtype=float)

    # 读取 GPS odomenu，计算 target yaw + 与当前外参下的 yaw_lidar，并得到误差
    samples: List[Sample] = []
    errors_current: List[float] = []

    t0 = None
    prev = None  # (t, x, y)

    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, t in bag.read_messages(topics=[args.gps_topic]):
            ts = _read_time(t, msg, args.time_source)
            if t0 is None:
                t0 = ts

            pos = msg.pose.pose.position
            x = float(pos.x)
            y = float(pos.y)

            speed = 0.0
            yaw_from_pos = None
            if prev is not None:
                dt = ts - prev[0]
                if dt > 1e-3:
                    dx = x - prev[1]
                    dy = y - prev[2]
                    speed = math.hypot(dx, dy) / dt
                    if speed > 1e-6:
                        yaw_from_pos = math.atan2(dy, dx)

            prev = (ts, x, y)

            if speed < args.min_speed:
                continue

            ori = msg.pose.pose.orientation
            yaw_gps_quat = _quat_to_yaw_rad_xyzw(float(ori.x), float(ori.y), float(ori.z), float(ori.w))
            if args.gps_heading_source == "quat":
                yaw_gps = yaw_gps_quat
            else:
                if yaw_from_pos is None:
                    continue
                yaw_gps = yaw_from_pos

            yaw_target = _wrap_rad(yaw_gps + target_offset)

            # nearest IMU
            idx = int(np.searchsorted(imu_t_np, ts))
            cand: List[int] = []
            if idx > 0:
                cand.append(idx - 1)
            if idx < len(imu_t_np):
                cand.append(idx)
            best = min(cand, key=lambda i: abs(imu_t_np[i] - ts))
            if abs(float(imu_t_np[best] - ts)) > args.sync_tol:
                continue

            qx, qy, qz, qw = imu_q[best]
            R_w_i = _quat_xyzw_to_rotmat(qx, qy, qz, qw)

            # extRPY: IMU->LiDAR，姿态输出使用 R_w_l = R_w_i * R_i_l = R_w_i * extRPY^T
            R_w_l_current = R_w_i @ ext_rpy_old.T
            yaw_lidar_current = _wrap_rad(_yaw_from_rotmat(R_w_l_current))

            err_current = _wrap_rad(yaw_lidar_current - yaw_target)
            errors_current.append(err_current)

            # 先占位，等校准后再填
            samples.append(
                Sample(
                    t=ts,
                    t_rel=ts - t0,
                    speed=speed,
                    yaw_target=yaw_target,
                    yaw_lidar_current=yaw_lidar_current,
                    err_current=err_current,
                    yaw_lidar_calib=0.0,
                    err_calib=0.0,
                )
            )

            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break

    if len(samples) < 20:
        print(f"[ERROR] 有效样本过少: {len(samples)} (检查 --min-speed / --time-source / --gps-topic)", file=sys.stderr)
        return 2

    err_current_np = np.asarray(errors_current, dtype=float)
    delta = _circular_mean(err_current_np)
    delta_std = _circular_std(err_current_np)

    # 计算新外参：仅对 output frame(LiDAR) 施加绕Z微调
    Rz_delta = _rotz(delta)
    ext_rot_new = Rz_delta @ ext_rot_old
    ext_rpy_new = Rz_delta @ ext_rpy_old

    # 重新计算校准后的误差
    errors_calib: List[float] = []
    for s in samples:
        # yaw_new = yaw_old - delta (对于纯Z旋转成立；这里直接按误差定义重新计算更稳妥)
        yaw_lidar_calib = _wrap_rad(s.yaw_lidar_current - delta)
        err_calib = _wrap_rad(yaw_lidar_calib - s.yaw_target)
        s.yaw_lidar_calib = yaw_lidar_calib
        s.err_calib = err_calib
        errors_calib.append(err_calib)

    err_calib_np = np.asarray(errors_calib, dtype=float)

    def stats_block(err: np.ndarray) -> Dict[str, float]:
        err_deg = np.degrees(err)
        abs_deg = np.abs(err_deg)
        return {
            "mean_deg": float(np.mean(err_deg)),
            "std_deg": float(np.std(err_deg)),
            "rmse_deg": float(math.sqrt(float(np.mean(err_deg**2)))),
            "median_abs_deg": float(np.median(abs_deg)),
            "p95_abs_deg": float(np.percentile(abs_deg, 95)),
            "max_abs_deg": float(np.max(abs_deg)),
            "n": int(err_deg.size),
        }

    summary = {
        "bag": bag_path,
        "params": params_path,
        "topics": {"imu": args.imu_topic, "gps": args.gps_topic, "lidar": args.lidar_topic},
        "time_source": args.time_source,
        "gps_heading_source": args.gps_heading_source,
        "min_speed": args.min_speed,
        "sync_tol": args.sync_tol,
        "lidar_forward_axis": forward_axis,
        "lidar_forward_ratios": forward_ratios,
        "target_offset_deg": float(math.degrees(target_offset)),
        "delta_deg": float(math.degrees(delta)),
        "delta_circular_std_deg": float(math.degrees(delta_std)),
        "ext_rot_old_row_major": _format_rot_row_major(ext_rot_old),
        "ext_rpy_old_row_major": _format_rot_row_major(ext_rpy_old),
        "ext_rot_new_row_major": _format_rot_row_major(ext_rot_new),
        "ext_rpy_new_row_major": _format_rot_row_major(ext_rpy_new),
        "error_current": stats_block(err_current_np),
        "error_calibrated": stats_block(err_calib_np),
    }

    # 保存 CSV
    csv_path = out_dir / "samples.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "t_rel,t,speed,yaw_target_deg,yaw_lidar_current_deg,err_current_deg,"
            "yaw_lidar_calib_deg,err_calib_deg\n"
        )
        for s in samples:
            f.write(
                "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    s.t_rel,
                    s.t,
                    s.speed,
                    math.degrees(s.yaw_target),
                    math.degrees(s.yaw_lidar_current),
                    math.degrees(s.err_current),
                    math.degrees(s.yaw_lidar_calib),
                    math.degrees(s.err_calib),
                )
            )

    # 保存 summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 绘图：误差-时间
    t_rel = np.array([s.t_rel for s in samples], dtype=float)
    err_cur_deg = np.degrees(np.array([s.err_current for s in samples], dtype=float))
    err_cal_deg = np.degrees(np.array([s.err_calib for s in samples], dtype=float))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_rel, err_cur_deg, label="Current params.yaml", linewidth=1.0)
    ax.plot(t_rel, err_cal_deg, label="Calibrated (yaw delta applied)", linewidth=1.0)
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_title("IMU->LiDAR yaw residual vs time")
    ax.set_xlabel("t (s, relative)")
    ax.set_ylabel("yaw residual (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "yaw_error_vs_time.png", dpi=200)
    plt.close(fig)

    # 绘图：误差直方图
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.hist(err_cur_deg, bins=60, alpha=0.8)
    ax.set_title("Current residual (deg)")
    ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(1, 2, 2)
    ax.hist(err_cal_deg, bins=60, alpha=0.8)
    ax.set_title("Calibrated residual (deg)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "yaw_error_hist.png", dpi=200)
    plt.close(fig)

    # 滑窗估计 delta(t)
    if args.window_sec > 0:
        win = float(args.window_sec)
        delta_ts = []
        delta_deg_series = []
        for i in range(len(samples)):
            t_i = samples[i].t
            t_start = t_i - win / 2.0
            t_end = t_i + win / 2.0
            errs = [s.err_current for s in samples if t_start <= s.t <= t_end]
            if len(errs) < 10:
                continue
            delta_i = _circular_mean(np.asarray(errs, dtype=float))
            delta_ts.append(samples[i].t_rel)
            delta_deg_series.append(math.degrees(delta_i))

        if delta_ts:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(delta_ts, delta_deg_series, linewidth=1.0)
            ax.axhline(math.degrees(delta), color="k", linewidth=0.8, alpha=0.5, label="global delta")
            ax.set_title(f"Estimated yaw delta over time (window={win:.1f}s)")
            ax.set_xlabel("t (s, relative)")
            ax.set_ylabel("delta (deg)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "delta_window_vs_time.png", dpi=200)
            plt.close(fig)

    # 推荐参数输出
    def fmt_mat(m: np.ndarray) -> str:
        v = _format_rot_row_major(m, decimals=6)
        return "[{}, {}, {},\n                 {}, {}, {},\n                 {}, {}, {}]".format(*v)

    rec_txt = (
        "=== Recommended extrinsics (apply to params.yaml) ===\n"
        f"lidar_forward_axis: {forward_axis}\n"
        f"target_offset_deg: {math.degrees(target_offset):.3f}\n"
        f"yaw_delta_deg (applied to LiDAR/output frame): {math.degrees(delta):.6f}\n\n"
        "extrinsicRot:\n"
        f"{fmt_mat(ext_rot_new)}\n\n"
        "extrinsicRPY:\n"
        f"{fmt_mat(ext_rpy_new)}\n"
    )
    with open(out_dir / "recommended.txt", "w", encoding="utf-8") as f:
        f.write(rec_txt)

    print(rec_txt)
    print("=== Error stats ===")
    print("Current:", summary["error_current"])
    print("Calibrated:", summary["error_calibrated"])
    print(f"[INFO] Outputs saved to: {out_dir}")

    if args.write_params:
        _patch_params_yaml_inplace(params_path, ext_rot_new, ext_rpy_new)
        print(f"[INFO] Updated params.yaml: {params_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
