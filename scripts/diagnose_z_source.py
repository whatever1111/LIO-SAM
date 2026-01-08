#!/usr/bin/env python3
"""
离线诊断：找出“飞了”的 Z 到底来自哪条数据。

典型现象：
- useGpsElevation=false 时，mapOptmization.cpp 会把 GPS 因子里的 z 直接替换为当前 LIO 的 z，
  所以日志里看到的 “GPS Factor z 很大” 往往并不是 GNSS 的 z 坏了，而是 LIO 自己的 z 已经漂了。

本脚本读取一个评估/录制 bag（比如 /tmp/trajectory_evaluation.bag），对以下话题的 z 做统计和跳变检测：
  - /odometry/gps
  - /lio_sam/mapping/odometry
  - /lio_sam/mapping/odometry_incremental
  - /odometry/imu
  - /odometry/imu_incremental
并可关联 /lio_sam/mapping/odometry_incremental_status 的 is_degenerate 标志。

用法：
  python3 scripts/diagnose_z_source.py /tmp/trajectory_evaluation.bag --out /tmp/z_diag

提示：
  如果你的评估 bag 还没有 /odometry/imu(_incremental)，请先更新并使用 scripts/record_trajectory.py 重新录一次。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rosbag  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Failed to import rosbag. Did you source ROS? Error: {e}")


DEFAULT_TOPICS: Dict[str, str] = {
    "gps": "/odometry/gps",
    "lio_map": "/lio_sam/mapping/odometry",
    "lio_incre": "/lio_sam/mapping/odometry_incremental",
    "imu_fused": "/odometry/imu",
    "imu_incre": "/odometry/imu_incremental",
}

STATUS_TOPIC = "/lio_sam/mapping/odometry_incremental_status"


@dataclass
class Series:
    name: str
    topic: str
    t: np.ndarray  # seconds
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def t0(self) -> float:
        return float(self.t[0]) if self.n else float("nan")


def _load_odometry_series(bag: "rosbag.Bag", name: str, topic: str) -> Optional[Series]:
    t_list: List[float] = []
    x_list: List[float] = []
    y_list: List[float] = []
    z_list: List[float] = []

    for _, msg, t in bag.read_messages(topics=[topic]):
        # Prefer header stamp; fall back to bag time if needed.
        try:
            stamp = msg.header.stamp.to_sec()
        except Exception:
            stamp = t.to_sec()

        try:
            p = msg.pose.pose.position
            x_list.append(float(p.x))
            y_list.append(float(p.y))
            z_list.append(float(p.z))
            t_list.append(float(stamp))
        except Exception:
            continue

    if not t_list:
        return None

    t_arr = np.asarray(t_list, dtype=float)
    x_arr = np.asarray(x_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)
    z_arr = np.asarray(z_list, dtype=float)
    return Series(name=name, topic=topic, t=t_arr, x=x_arr, y=y_arr, z=z_arr)


def _load_degenerate_flags(bag: "rosbag.Bag") -> Dict[float, bool]:
    flags: Dict[float, bool] = {}
    for _, msg, _t in bag.read_messages(topics=[STATUS_TOPIC]):
        try:
            stamp = float(msg.header.stamp.to_sec())
            flags[stamp] = bool(msg.is_degenerate)
        except Exception:
            continue
    return flags


def _baseline_z(series: Series, baseline_sec: float) -> float:
    if series.n == 0:
        return float("nan")
    t0 = series.t0
    mask = series.t <= (t0 + float(baseline_sec))
    if not np.any(mask):
        return float(np.median(series.z))
    return float(np.median(series.z[mask]))


def _first_exceed(series: Series, baseline: float, thr: float) -> Optional[Tuple[float, float, float]]:
    """
    Returns: (t, rel_time, z) for first |z-baseline| > thr
    """
    if series.n == 0:
        return None
    dz = np.abs(series.z - float(baseline))
    idx = np.where(dz > float(thr))[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    return float(series.t[i]), float(series.t[i] - series.t0), float(series.z[i])


def _top_jumps(series: Series, k: int) -> List[Tuple[float, float, float, float, float]]:
    """
    Returns list of (t, rel_time, z0, z1, dz) sorted by |dz| desc.
    """
    if series.n < 2:
        return []
    dz = np.diff(series.z)
    idx = np.argsort(np.abs(dz))[::-1][: int(k)]
    out: List[Tuple[float, float, float, float, float]] = []
    for i in idx:
        i = int(i)
        out.append(
            (
                float(series.t[i]),
                float(series.t[i] - series.t0),
                float(series.z[i]),
                float(series.z[i + 1]),
                float(dz[i]),
            )
        )
    return out


def _print_series_report(series: Series, baseline_sec: float, thr: float, deg_flags: Optional[Dict[float, bool]] = None) -> None:
    base = _baseline_z(series, baseline_sec)
    z = series.z
    print(f"\n[{series.name}] topic={series.topic}")
    print(f"  N={series.n}  t=[{series.t[0]:.3f} -> {series.t[-1]:.3f}]  dt={series.t[-1]-series.t[0]:.3f}s")
    print(f"  z: min={z.min():.3f}  max={z.max():.3f}  median={np.median(z):.3f}  p95={np.percentile(z,95):.3f}  std={z.std():.3f}")
    print(f"  baseline(z, first {baseline_sec:.1f}s) = {base:.3f}")

    first = _first_exceed(series, base, thr)
    if first is None:
        print(f"  first |z-baseline|>{thr:.3f} : (none)")
    else:
        t_abs, t_rel, z_val = first
        deg_txt = ""
        if deg_flags is not None and series.name in ("lio_map", "lio_incre"):
            deg_txt = f", degenerate={deg_flags.get(t_abs, False)}"
        print(f"  first |z-baseline|>{thr:.3f} : rel={t_rel:.3f}s  t={t_abs:.3f}  z={z_val:.3f}{deg_txt}")

    jumps = _top_jumps(series, k=5)
    if jumps:
        print("  top |dz| jumps:")
        for t_abs, t_rel, z0, z1, dz in jumps:
            deg_txt = ""
            if deg_flags is not None and series.name in ("lio_map", "lio_incre"):
                deg_txt = f", degenerate={deg_flags.get(t_abs, False)}"
            print(f"    rel={t_rel:8.3f}s  z: {z0:+9.3f} -> {z1:+9.3f}  dz={dz:+9.3f}{deg_txt}")


def _maybe_plot(series_list: List[Series], deg_flags: Dict[float, bool], out_dir: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    if not series_list:
        return None

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    t0 = min(s.t0 for s in series_list if s.n)

    for s in series_list:
        if s.n == 0:
            continue
        ax.plot(s.t - t0, s.z, label=s.name, linewidth=1.0)

    # Shade degenerate intervals (based on incremental status stamps).
    if deg_flags:
        stamps = np.array(sorted(deg_flags.keys()), dtype=float)
        vals = np.array([deg_flags[k] for k in stamps], dtype=bool)
        if stamps.size > 0:
            in_deg = False
            start = None
            for ts, v in zip(stamps, vals):
                if v and not in_deg:
                    in_deg = True
                    start = ts
                if (not v) and in_deg:
                    in_deg = False
                    if start is not None:
                        ax.axvspan(start - t0, ts - t0, color="red", alpha=0.12)
                        start = None
            if in_deg and start is not None:
                ax.axvspan(start - t0, stamps[-1] - t0, color="red", alpha=0.12)

    ax.set_xlabel("Time (s, relative)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Z Time Series (red shade = degenerate)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "z_timeseries.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline Z drift source diagnosis from a recorded trajectory bag.")
    ap.add_argument("bag", help="Trajectory bag (e.g. /tmp/trajectory_evaluation.bag)")
    ap.add_argument("--baseline-sec", type=float, default=10.0, help="Baseline window length in seconds")
    ap.add_argument("--z-threshold", type=float, default=2.0, help="Threshold for |z-baseline| to flag drift")
    ap.add_argument("--out", default="", help="Output directory for plots (optional)")
    args = ap.parse_args()

    bag_path = os.path.expanduser(args.bag)
    if not os.path.exists(bag_path):
        raise SystemExit(f"Bag not found: {bag_path}")

    print("=" * 70)
    print("Z Drift Source Diagnosis")
    print("=" * 70)
    print(f"bag: {bag_path}")
    print(f"baseline_sec: {args.baseline_sec}")
    print(f"z_threshold: {args.z_threshold}")

    with rosbag.Bag(bag_path, "r", allow_unindexed=True) as bag:
        deg_flags = _load_degenerate_flags(bag)

        series_list: List[Series] = []
        for name, topic in DEFAULT_TOPICS.items():
            s = _load_odometry_series(bag, name=name, topic=topic)
            if s is None:
                continue
            series_list.append(s)

    if not series_list:
        print("No supported odometry topics found in bag.")
        return 2

    found_topics = {s.name for s in series_list}
    print("found topics:", ", ".join(sorted(found_topics)))
    if STATUS_TOPIC not in deg_flags and deg_flags:
        # deg_flags keys exist only if status topic present; this is a guard.
        pass

    for s in series_list:
        _print_series_report(s, baseline_sec=args.baseline_sec, thr=args.z_threshold, deg_flags=deg_flags if deg_flags else None)

    # Quick heuristic conclusion.
    print("\n" + "-" * 70)
    gps = next((s for s in series_list if s.name == "gps"), None)
    lio = next((s for s in series_list if s.name == "lio_map"), None)
    if gps is not None and lio is not None:
        gps_range = float(gps.z.max() - gps.z.min())
        lio_range = float(lio.z.max() - lio.z.min())
        print(f"heuristic: gps z range={gps_range:.3f} m, lio_map z range={lio_range:.3f} m")
        if gps_range < 1.0 and lio_range > 5.0:
            print("likely: GNSS 的 z 正常，异常来自 LIO-SAM 自己的 z（LiDAR/IMU 估计漂移或跳变）")
        elif gps_range > 5.0:
            print("likely: GNSS z 本身有明显跳变（先检查 /odometry/gps 的 z 与 cov）")
        else:
            print("likely: z 变化不大或两者都正常（若仍“飞”，可能是 XY/yaw 或时间同步问题）")
    else:
        print("heuristic: 缺少 gps 或 lio_map 话题，无法做快速结论（请确保录到了 /odometry/gps 与 /lio_sam/mapping/odometry）")

    if args.out:
        out_dir = os.path.expanduser(args.out)
        plot_path = _maybe_plot(series_list, deg_flags, out_dir)
        if plot_path:
            print(f"saved plot: {plot_path}")
        else:
            print("plot skipped (matplotlib not available)")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

