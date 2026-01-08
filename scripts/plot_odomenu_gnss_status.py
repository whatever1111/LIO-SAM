#!/usr/bin/env python3
"""
Offline plotter for Fixposition /fixposition/fpa/odomenu position and GNSS status.

What it produces (PNG):
  - odomenu_position_full.png: x/y/z/norm over full bag time
  - gnss_status_full.png: gnss1/gnss2 status and degraded flag over time
  - odomenu_jump_zoom_<N>.png: zoomed around each detected jump (position + GNSS)

This is designed for the "jump" investigation: when odomenu suddenly changes by
hundreds/thousands of meters (usually due to ENU origin reset / invalid GNSS).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Matplotlib backend: safe in headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rosbag


@dataclass
class JumpEvent:
    idx: int
    t_bag: float
    dt: float
    dist: float
    prev_xyz: Tuple[float, float, float]
    curr_xyz: Tuple[float, float, float]
    gnss1: int
    gnss2: int


def _interp_nd(t_src: np.ndarray, y_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """
    1D time interpolation for NxD arrays.
    - t_src: (N,)
    - y_src: (N,D)
    - t_query: (M,)
    Returns: (M,D)
    """
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    if t_src.size == 0 or y_src.size == 0 or t_query.size == 0:
        return np.zeros((0, y_src.shape[1] if y_src.ndim == 2 else 0), dtype=float)
    if y_src.ndim != 2:
        raise ValueError(f"y_src must be 2D, got {y_src.ndim}D")
    out = np.zeros((t_query.shape[0], y_src.shape[1]), dtype=float)
    for i in range(y_src.shape[1]):
        out[:, i] = np.interp(t_query, t_src, y_src[:, i])
    return out


def _estimate_se2(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate rigid SE(2) transform mapping src_xy -> dst_xy.
    Returns: (R2 (2x2), t2 (2,), yaw_deg)
    """
    src_xy = np.asarray(src_xy, dtype=float)
    dst_xy = np.asarray(dst_xy, dtype=float)
    if src_xy.shape != dst_xy.shape or src_xy.ndim != 2 or src_xy.shape[1] != 2:
        raise ValueError(f"expected Nx2 arrays, got src={src_xy.shape}, dst={dst_xy.shape}")
    if src_xy.shape[0] < 3:
        raise ValueError("need at least 3 points to estimate SE2")

    src_mean = src_xy.mean(axis=0)
    dst_mean = dst_xy.mean(axis=0)
    S = src_xy - src_mean
    D = dst_xy - dst_mean
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - (R @ src_mean)
    yaw = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return R, t, yaw


def _as_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _compute_degraded(gnss1: int, gnss2: int, min_fix_for_good: int, degraded_if_any_below_min: bool) -> bool:
    if degraded_if_any_below_min:
        return (gnss1 < min_fix_for_good) or (gnss2 < min_fix_for_good)
    return (gnss1 < min_fix_for_good) and (gnss2 < min_fix_for_good)


def _load_odomenu_series(
    bag_path: str,
    topic: str,
    jump_threshold_m: float,
    min_fix_for_good: int,
    degraded_if_any_below_min: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[JumpEvent]]:
    """
    Returns:
      t (N,), xyz (N,3), norm (N,), gnss (N,2), degraded (N,), jumps (list)
    """
    t_list: List[float] = []
    xyz_list: List[Tuple[float, float, float]] = []
    gnss_list: List[Tuple[int, int]] = []
    degraded_list: List[bool] = []
    jumps: List[JumpEvent] = []

    prev = None  # (t, x, y, z)

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            t_bag = float(t.to_sec())
            p = msg.pose.pose.position
            x = _safe_float(getattr(p, "x", 0.0), 0.0)
            y = _safe_float(getattr(p, "y", 0.0), 0.0)
            z = _safe_float(getattr(p, "z", 0.0), 0.0)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue

            gnss1 = _as_int(getattr(msg, "gnss1_status", 0), 0)
            gnss2 = _as_int(getattr(msg, "gnss2_status", 0), 0)
            degraded = _compute_degraded(gnss1, gnss2, min_fix_for_good, degraded_if_any_below_min)

            idx = len(t_list)
            if prev is not None:
                dt = t_bag - prev[0]
                dx = x - prev[1]
                dy = y - prev[2]
                dz = z - prev[3]
                dist = float(math.sqrt(dx * dx + dy * dy + dz * dz))
                if dist >= jump_threshold_m:
                    jumps.append(
                        JumpEvent(
                            idx=idx,
                            t_bag=t_bag,
                            dt=float(dt),
                            dist=float(dist),
                            prev_xyz=(float(prev[1]), float(prev[2]), float(prev[3])),
                            curr_xyz=(float(x), float(y), float(z)),
                            gnss1=gnss1,
                            gnss2=gnss2,
                        )
                    )

            t_list.append(t_bag)
            xyz_list.append((x, y, z))
            gnss_list.append((gnss1, gnss2))
            degraded_list.append(bool(degraded))
            prev = (t_bag, x, y, z)

    t_arr = np.asarray(t_list, dtype=float)
    xyz_arr = np.asarray(xyz_list, dtype=float)
    norm_arr = np.linalg.norm(xyz_arr, axis=1) if xyz_arr.size else np.zeros((0,), dtype=float)
    gnss_arr = np.asarray(gnss_list, dtype=int)
    degraded_arr = np.asarray(degraded_list, dtype=bool)
    return t_arr, xyz_arr, norm_arr, gnss_arr, degraded_arr, jumps


def _load_odom_series(bag_path: str, topic: str) -> Tuple[np.ndarray, np.ndarray]:
    t_list: List[float] = []
    xyz_list: List[Tuple[float, float, float]] = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            t_bag = float(t.to_sec())
            p = msg.pose.pose.position
            x = _safe_float(getattr(p, "x", 0.0), 0.0)
            y = _safe_float(getattr(p, "y", 0.0), 0.0)
            z = _safe_float(getattr(p, "z", 0.0), 0.0)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            t_list.append(t_bag)
            xyz_list.append((x, y, z))
    return np.asarray(t_list, dtype=float), np.asarray(xyz_list, dtype=float)


def _plot_position_full(t: np.ndarray, xyz: np.ndarray, norm: np.ndarray, out_png: str, title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    labels = ["x (m)", "y (m)", "z (m)", "||p|| (m)"]
    series = [xyz[:, 0], xyz[:, 1], xyz[:, 2], norm]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ax, y, lab, c in zip(axes, series, labels, colors):
        ax.plot(t - t[0], y, color=c, linewidth=1.0)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("bag time (s, relative)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_gnss_full(t: np.ndarray, gnss: np.ndarray, degraded: np.ndarray, out_png: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    axes[0].step(t - t[0], gnss[:, 0], where="post", label="gnss1_status", linewidth=1.2)
    axes[0].step(t - t[0], gnss[:, 1], where="post", label="gnss2_status", linewidth=1.2)
    axes[0].set_ylabel("fix type (int)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].step(t - t[0], degraded.astype(int), where="post", color="tab:red", linewidth=1.5)
    axes[1].set_ylabel("degraded (0/1)")
    axes[1].set_xlabel("bag time (s, relative)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("GNSS status from /fixposition/fpa/odomenu (full)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_xy_aligned(
    t_ref: np.ndarray,
    ref_xy: np.ndarray,
    t_src: np.ndarray,
    src_xy: np.ndarray,
    out_png: str,
    jump_times: Optional[List[float]] = None,
    title: str = "XY comparison",
) -> None:
    """
    Plot XY trajectories overlayed. ref_xy is interpolated onto t_src timestamps so both are consistent.
    """
    t_start = max(float(t_ref[0]), float(t_src[0]))
    t_end = min(float(t_ref[-1]), float(t_src[-1]))
    mask = (t_src >= t_start) & (t_src <= t_end)
    if not np.any(mask):
        return

    t_common = t_src[mask]
    ref_interp = _interp_nd(t_ref, ref_xy, t_common)
    src_common = src_xy[mask]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.plot(ref_interp[:, 0], ref_interp[:, 1], "-", linewidth=2.0, alpha=0.75, label="odom (/odom) XY")
    ax.plot(src_common[:, 0], src_common[:, 1], "-", linewidth=2.0, alpha=0.75, label="fpa odomenu (aligned) XY")

    if jump_times:
        for i, jt in enumerate(jump_times, 1):
            if jt < t_start or jt > t_end:
                continue
            idx = int(np.argmin(np.abs(t_common - jt)))
            ax.plot([src_common[idx, 0]], [src_common[idx, 1]], "ko", markersize=4, alpha=0.8)
            ax.text(src_common[idx, 0], src_common[idx, 1], f"jump{i}", fontsize=9)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_xy_error(
    t_common: np.ndarray,
    err_xy: np.ndarray,
    out_png: str,
    jump_times: Optional[List[float]] = None,
    t0: Optional[float] = None,
    title: str = "XY error",
) -> None:
    if t0 is None:
        t0 = float(t_common[0])
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(t_common - t0, err_xy, "b-", linewidth=1.5)
    ax.set_xlabel("bag time (s, relative)")
    ax.set_ylabel("||odom - fpa_aligned|| (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if jump_times:
        for jt in jump_times:
            ax.axvline(float(jt) - t0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_jump_zoom(
    t: np.ndarray,
    xyz: np.ndarray,
    gnss: np.ndarray,
    degraded: np.ndarray,
    jump: JumpEvent,
    window_s: float,
    out_png: str,
) -> None:
    t0 = jump.t_bag - float(window_s)
    t1 = jump.t_bag + float(window_s)
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return

    tw = t[mask]
    xyzw = xyz[mask]
    gnssw = gnss[mask]
    degg = degraded[mask]

    # Position + GNSS in one figure for quick inspection.
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    labels = ["x (m)", "y (m)", "z (m)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, (lab, c) in enumerate(zip(labels, colors)):
        axes[i].plot(tw - tw[0], xyzw[:, i], color=c, linewidth=1.2)
        axes[i].axvline(jump.t_bag - tw[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)
        axes[i].set_ylabel(lab)
        axes[i].grid(True, alpha=0.3)

    axes[3].step(tw - tw[0], gnssw[:, 0], where="post", label="gnss1_status", linewidth=1.2)
    axes[3].step(tw - tw[0], gnssw[:, 1], where="post", label="gnss2_status", linewidth=1.2)
    axes[3].axvline(jump.t_bag - tw[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[3].set_ylabel("fix type")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    axes[4].step(tw - tw[0], degg.astype(int), where="post", color="tab:red", linewidth=1.5)
    axes[4].axvline(jump.t_bag - tw[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[4].set_ylabel("degraded")
    axes[4].set_xlabel("bag time (s, relative in window)")
    axes[4].grid(True, alpha=0.3)

    fig.suptitle(
        f"Jump @ +{jump.t_bag - t[0]:.3f}s  dist={jump.dist:.1f}m  dt={jump.dt:.3f}s  gnss=({jump.gnss1},{jump.gnss2})",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_jump_zoom_compare(
    t_g: np.ndarray,
    xyz_g: np.ndarray,
    gnss: np.ndarray,
    degraded: np.ndarray,
    jump: JumpEvent,
    window_s: float,
    t_odom: Optional[np.ndarray],
    xyz_odom: Optional[np.ndarray],
    out_png: str,
) -> None:
    """
    Compare /odom vs /fixposition/fpa/odomenu within a window around a jump.
    Uses delta-to-window-start so both curves remain visible even if absolute scales differ.
    """
    t0 = jump.t_bag - float(window_s)
    t1 = jump.t_bag + float(window_s)
    mask = (t_g >= t0) & (t_g <= t1)
    if not np.any(mask):
        return

    tw = t_g[mask]
    gwin = xyz_g[mask]
    gnssw = gnss[mask]
    degg = degraded[mask]

    gdelta = gwin - gwin[0]

    odom_delta = None
    if t_odom is not None and xyz_odom is not None and t_odom.size >= 2 and xyz_odom.size > 0:
        t_start = max(float(t_odom[0]), float(tw[0]))
        t_end = min(float(t_odom[-1]), float(tw[-1]))
        if t_end > t_start:
            m2 = (tw >= t_start) & (tw <= t_end)
            if np.any(m2):
                tw2 = tw[m2]
                odom_interp = _interp_nd(t_odom, xyz_odom, tw2)
                odom_delta = (tw2, odom_interp - odom_interp[0], m2)

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    labels = ["Δx (m)", "Δy (m)", "Δz (m)"]
    colors_g = ["tab:blue", "tab:orange", "tab:green"]
    colors_o = ["tab:purple", "tab:brown", "tab:gray"]

    for i in range(3):
        axes[i].plot(tw - tw[0], gdelta[:, i], color=colors_g[i], linewidth=1.5, label="fpa odomenu Δ")
        if odom_delta is not None:
            tw2, od, _m2 = odom_delta
            axes[i].plot(tw2 - tw[0], od[:, i], color=colors_o[i], linewidth=1.5, alpha=0.85, label="/odom Δ")
        axes[i].axvline(jump.t_bag - tw[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="best")

    axes[3].step(tw - tw[0], gnssw[:, 0], where="post", label="gnss1_status", linewidth=1.2)
    axes[3].step(tw - tw[0], gnssw[:, 1], where="post", label="gnss2_status", linewidth=1.2)
    axes[3].step(tw - tw[0], degg.astype(int), where="post", label="degraded(0/1)", linewidth=1.2, color="tab:red")
    axes[3].axvline(jump.t_bag - tw[0], color="k", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[3].set_ylabel("GNSS/degraded")
    axes[3].set_xlabel("bag time (s, relative in window)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle(
        f"/odom vs fpa odomenu around jump @ +{jump.t_bag - t_g[0]:.3f}s  dist={jump.dist:.1f}m",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot /fixposition/fpa/odomenu position and GNSS status from a rosbag.")
    parser.add_argument("--bag", required=True, help="rosbag path")
    parser.add_argument("--topic", default="/fixposition/fpa/odomenu", help="topic to read (default: /fixposition/fpa/odomenu)")
    parser.add_argument("--odom-topic", default="/odom", help="odom topic to compare (default: /odom)")
    parser.add_argument("--out-dir", default="/tmp/odomenu_gnss_plots", help="output directory")
    parser.add_argument("--jump-threshold-m", type=float, default=1000.0, help="detect jumps larger than this (meters)")
    parser.add_argument("--window-s", type=float, default=20.0, help="zoom window half-size around each jump (seconds)")
    parser.add_argument(
        "--align-max-norm-m",
        type=float,
        default=1000.0,
        help="use only odomenu points with ||p|| below this for /odom alignment (meters)",
    )
    parser.add_argument("--min-fix-for-good", type=int, default=8, help="threshold for 'good' GNSS (Fixposition consts, e.g. 7=float, 8=fixed)")
    parser.add_argument(
        "--degraded-if-any-below-min",
        action="store_true",
        default=True,
        help="degraded if either antenna is below min_fix_for_good (default true)",
    )
    parser.add_argument(
        "--degraded-if-both-below-min",
        dest="degraded_if_any_below_min",
        action="store_false",
        help="degraded only if both antennas are below min_fix_for_good",
    )
    args = parser.parse_args()

    bag_path = os.path.expanduser(args.bag)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t, xyz, norm, gnss, degraded, jumps = _load_odomenu_series(
        bag_path=bag_path,
        topic=args.topic,
        jump_threshold_m=float(args.jump_threshold_m),
        min_fix_for_good=int(args.min_fix_for_good),
        degraded_if_any_below_min=bool(args.degraded_if_any_below_min),
    )

    if t.size == 0:
        raise SystemExit(f"No messages read from {args.topic} in bag: {bag_path}")

    _plot_position_full(
        t,
        xyz,
        norm,
        os.path.join(out_dir, "odomenu_position_full.png"),
        title="/fixposition/fpa/odomenu position (full)",
    )
    _plot_gnss_full(t, gnss, degraded, os.path.join(out_dir, "gnss_status_full.png"))

    # Optional /odom comparison
    t_odom = None
    xyz_odom = None
    try:
        t_odom, xyz_odom = _load_odom_series(bag_path, args.odom_topic)
        if t_odom.size == 0:
            t_odom, xyz_odom = None, None
            print(f"Warning: no messages read from {args.odom_topic}; skipping /odom plots")
    except Exception as e:
        t_odom, xyz_odom = None, None
        print(f"Warning: failed to read {args.odom_topic}: {e}; skipping /odom plots")

    if t_odom is not None and xyz_odom is not None:
        odom_norm = np.linalg.norm(xyz_odom, axis=1) if xyz_odom.size else np.zeros((0,), dtype=float)
        _plot_position_full(
            t_odom,
            xyz_odom,
            odom_norm,
            os.path.join(out_dir, "odom_position_full.png"),
            title=f"{args.odom_topic} position (full)",
        )

        # Align odomenu XY -> /odom XY using only non-outlier points (by ||p||).
        t_start = max(float(t[0]), float(t_odom[0]))
        t_end = min(float(t[-1]), float(t_odom[-1]))
        mask_common = (t >= t_start) & (t <= t_end)
        if np.any(mask_common):
            t_common = t[mask_common]
            odom_xy_interp = _interp_nd(t_odom, xyz_odom[:, :2], t_common)
            odomenu_xy = xyz[mask_common, :2]

            good_align = norm[mask_common] < float(args.align_max_norm_m)
            if int(good_align.sum()) >= 3:
                try:
                    R2, t2, yaw_deg = _estimate_se2(odomenu_xy[good_align], odom_xy_interp[good_align])
                    odomenu_xy_aligned = (odomenu_xy @ R2.T) + t2
                    jump_times = [j.t_bag for j in jumps]
                    _plot_xy_aligned(
                        t_ref=t_common,
                        ref_xy=odom_xy_interp,
                        t_src=t_common,
                        src_xy=odomenu_xy_aligned,
                        out_png=os.path.join(out_dir, "odom_vs_odomenu_xy_aligned.png"),
                        jump_times=jump_times,
                        title=f"/odom vs fpa odomenu (aligned SE2)  yaw={yaw_deg:+.2f}deg",
                    )
                    err_xy = np.linalg.norm(odom_xy_interp - odomenu_xy_aligned, axis=1)
                    _plot_xy_error(
                        t_common=t_common,
                        err_xy=err_xy,
                        out_png=os.path.join(out_dir, "odom_vs_odomenu_xy_error.png"),
                        jump_times=jump_times,
                        t0=float(t[0]),
                        title="XY error: /odom vs fpa odomenu (aligned)",
                    )
                except Exception as e:
                    print(f"Warning: alignment plot skipped (SE2 fit failed): {e}")

    # Also write a short text summary for convenience.
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"bag: {bag_path}\n")
        f.write(f"topic: {args.topic}\n")
        f.write(f"odom_topic: {args.odom_topic}\n")
        f.write(f"points: {t.size}\n")
        f.write(f"time range (bag): {t[0]:.6f} -> {t[-1]:.6f}  dt={t[-1]-t[0]:.3f}s\n")
        f.write(f"jump_threshold_m: {args.jump_threshold_m}\n")
        f.write(f"align_max_norm_m: {args.align_max_norm_m}\n")
        f.write(f"min_fix_for_good: {args.min_fix_for_good}\n")
        f.write(f"degraded_if_any_below_min: {args.degraded_if_any_below_min}\n")
        f.write(f"detected jumps: {len(jumps)}\n")
        for i, j in enumerate(jumps, 1):
            f.write(
                f"  {i}: idx={j.idx} t={j.t_bag:.6f} (+{j.t_bag-t[0]:.3f}s) dt={j.dt:.3f}s dist={j.dist:.3f} "
                f"prev={j.prev_xyz} curr={j.curr_xyz} gnss=({j.gnss1},{j.gnss2})\n"
            )

    for i, j in enumerate(jumps, 1):
        out_png = os.path.join(out_dir, f"odomenu_jump_zoom_{i}.png")
        _plot_jump_zoom(t, xyz, gnss, degraded, j, window_s=float(args.window_s), out_png=out_png)

        out_png2 = os.path.join(out_dir, f"odom_vs_odomenu_jump_zoom_{i}.png")
        _plot_jump_zoom_compare(
            t_g=t,
            xyz_g=xyz,
            gnss=gnss,
            degraded=degraded,
            jump=j,
            window_s=float(args.window_s),
            t_odom=t_odom,
            xyz_odom=xyz_odom,
            out_png=out_png2,
        )

    print(f"Saved plots to: {out_dir}")
    print(f"  - {os.path.join(out_dir, 'odomenu_position_full.png')}")
    print(f"  - {os.path.join(out_dir, 'gnss_status_full.png')}")
    if t_odom is not None and xyz_odom is not None:
        print(f"  - {os.path.join(out_dir, 'odom_position_full.png')}")
        print(f"  - {os.path.join(out_dir, 'odom_vs_odomenu_xy_aligned.png')}")
        print(f"  - {os.path.join(out_dir, 'odom_vs_odomenu_xy_error.png')}")
    if jumps:
        print(f"  - {len(jumps)} jump zoom plot(s): odomenu_jump_zoom_*.png")
        print(f"  - {len(jumps)} jump compare plot(s): odom_vs_odomenu_jump_zoom_*.png")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
