#!/usr/bin/env python3
"""
Analyze GNSS holdout consistency (gps_train vs gps_test).

Goal:
  Explain why "non-degraded" holdout error can still be large by quantifying GNSS self-consistency.

Method:
  - Interpolate gps_train to gps_test timestamps.
  - Compute ||gps_test(t) - interp(gps_train)(t)|| over time.
  - Report statistics and list top outliers.

This does NOT require ground-truth. It tells you the noise/outlier level of the GNSS stream
even when the degraded flag says "good".
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Series:
    t: np.ndarray  # shape (N,)
    p: np.ndarray  # shape (N,3)


def _load_series(bag_path: str, topic: str) -> Series:
    try:
        import rosbag  # type: ignore
    except Exception as e:
        raise RuntimeError("rosbag python module not available; run inside a ROS environment") from e

    t_list: List[float] = []
    p_list: List[Tuple[float, float, float]] = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            t_list.append(t.to_sec())
            p_list.append(
                (
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    float(msg.pose.pose.position.z),
                )
            )
    if not t_list:
        return Series(t=np.zeros((0,), dtype=float), p=np.zeros((0, 3), dtype=float))
    t_arr = np.asarray(t_list, dtype=float)
    p_arr = np.asarray(p_list, dtype=float)
    order = np.argsort(t_arr)
    return Series(t=t_arr[order], p=p_arr[order])


def _load_degraded_timeline(bag_path: str, topic: str) -> Optional[Series]:
    try:
        import rosbag  # type: ignore
    except Exception:
        return None

    t_list: List[float] = []
    v_list: List[Tuple[float, float, float]] = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            t_list.append(t.to_sec())
            v = 1.0 if bool(getattr(msg, "data", False)) else 0.0
            v_list.append((v, 0.0, 0.0))
    if not t_list:
        return None
    t_arr = np.asarray(t_list, dtype=float)
    p_arr = np.asarray(v_list, dtype=float)
    order = np.argsort(t_arr)
    return Series(t=t_arr[order], p=p_arr[order])


def _interp_xyz(series: Series, t_samples: np.ndarray) -> np.ndarray:
    if series.t.size == 0 or t_samples.size == 0:
        return np.zeros((0, 3), dtype=float)
    out = np.zeros((t_samples.size, 3), dtype=float)
    for i in range(3):
        out[:, i] = np.interp(t_samples, series.t, series.p[:, i])
    return out


def _stats(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "n=0"
    return (
        f"n={x.size} mean={np.mean(x):.4f} rms={math.sqrt(float(np.mean(x*x))):.4f} "
        f"median={np.median(x):.4f} p90={np.percentile(x,90):.4f} p95={np.percentile(x,95):.4f} "
        f"max={np.max(x):.4f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Input bag containing gps_train/gps_test")
    ap.add_argument("--gps-train", default="/odometry/gps_train", help="Train GPS topic")
    ap.add_argument("--gps-test", default="/odometry/gps_test", help="Test GPS topic")
    ap.add_argument("--degraded-topic", default="/gnss_degraded", help="Optional degraded flag topic")
    ap.add_argument("--out", default=None, help="Optional output PNG path (plot)")
    ap.add_argument("--max-outliers", type=int, default=10, help="Print top-K outliers")
    args = ap.parse_args()

    train = _load_series(args.bag, args.gps_train)
    test = _load_series(args.bag, args.gps_test)
    if train.t.size == 0:
        print(f"[ERR] no messages on {args.gps_train}")
        return 2
    if test.t.size == 0:
        print(f"[ERR] no messages on {args.gps_test}")
        return 2

    # Only evaluate where interpolation is valid (inside train time span)
    mask = (test.t >= train.t[0]) & (test.t <= train.t[-1])
    t = test.t[mask]
    p_test = test.p[mask]
    p_train_i = _interp_xyz(train, t)
    e = p_test - p_train_i
    e_norm = np.linalg.norm(e, axis=1)

    print(f"bag: {args.bag}")
    print(f"train: {args.gps_train} ({train.t.size} msgs)")
    print(f"test:  {args.gps_test} ({test.t.size} msgs)")
    print(f"overlap_test_msgs: {t.size}")
    print("")
    print("||gps_test - interp(gps_train)|| (3D):", _stats(e_norm))

    # Degraded breakdown (if available)
    deg = _load_degraded_timeline(args.bag, args.degraded_topic)
    if deg is not None and deg.t.size > 0:
        # Use nearest degraded state (piecewise constant)
        idx = np.searchsorted(deg.t, t, side="right") - 1
        idx = np.clip(idx, 0, deg.t.size - 1)
        is_deg = (deg.p[idx, 0] > 0.5)
        if np.any(~is_deg):
            print("  degraded=0:", _stats(e_norm[~is_deg]))
        if np.any(is_deg):
            print("  degraded=1:", _stats(e_norm[is_deg]))

    # Outliers
    k = max(0, int(args.max_outliers))
    if k > 0 and e_norm.size > 0:
        order = np.argsort(-e_norm)  # descending
        print("")
        print(f"Top-{min(k, order.size)} outliers:")
        for i in order[:k]:
            print(
                f"  t={t[i]:.3f}  err={e_norm[i]:.3f}m  "
                f"dx={e[i,0]:+.3f} dy={e[i,1]:+.3f} dz={e[i,2]:+.3f}"
            )

    # Optional plot
    if args.out:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        t0 = float(t[0])
        plt.figure(figsize=(14, 5))
        plt.plot(t - t0, e_norm, "b-", linewidth=1.2, alpha=0.8, label="|gps_test - interp(gps_train)|")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.grid(True, alpha=0.3)
        plt.title("GNSS Holdout Self-Consistency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=200)
        print("")
        print(f"Saved plot: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

