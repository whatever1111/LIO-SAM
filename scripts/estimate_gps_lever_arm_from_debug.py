#!/usr/bin/env python3
"""
Estimate GNSS lever arm from LIO-SAM gps_factor_debug.csv.

If /odometry/gps represents the antenna position but the state represents the LiDAR/base position,
the residual (gps - lio) in the BODY frame should be approximately constant and equal to the
lever arm (LiDAR/base -> GNSS antenna).

This script estimates that constant offset using the debug CSV produced by:
  - /lio_sam/mapping/gps_factor_debug  (saved by scripts/evaluate_trajectory.py)
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple


def _wrap_deg(d: float) -> float:
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if q <= 0.0:
        return xs[0]
    if q >= 1.0:
        return xs[-1]
    k = (len(xs) - 1) * q
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _stats(name: str, xs: List[float]) -> str:
    if not xs:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(xs)} mean={sum(xs)/len(xs):.3f} "
        f"median={_quantile(xs, 0.5):.3f} p10={_quantile(xs, 0.1):.3f} p90={_quantile(xs, 0.9):.3f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to gps_factor_debug.csv")
    ap.add_argument("--include-degraded", action="store_true", help="Include gnss_degraded=1 rows")
    args = ap.parse_args()

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required for this script") from e

    df = pd.read_csv(args.csv)
    if "gps_pos_factor_added" not in df.columns:
        raise RuntimeError("csv does not look like gps_factor_debug.csv (missing gps_pos_factor_added)")

    # Focus on keyframes where a GPS position factor was actually added.
    sub = df[df["gps_pos_factor_added"] == 1].copy()
    if not args.include_degraded and "gnss_degraded" in sub.columns:
        sub = sub[sub["gnss_degraded"] == 0]

    if sub.empty:
        print("No GPS factors found after filtering.")
        return 0

    # Residual vector in map frame at decision time (pre-optimization for that keyframe)
    dx = (sub["gps_x"] - sub["pre_x"]).to_numpy()
    dy = (sub["gps_y"] - sub["pre_y"]).to_numpy()
    dz = (sub["gps_z"] - sub["pre_z"]).to_numpy()

    # Rotate map residual into BODY frame using yaw (approximation): body = Rz(yaw)^T * map
    yaw = (sub["pre_yaw_deg"]).to_numpy()
    dx_b: List[float] = []
    dy_b: List[float] = []
    dz_b: List[float] = []
    r_norm: List[float] = []
    for x, y, z, yaw_deg in zip(dx, dy, dz, yaw):
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(yaw_deg)):
            continue
        th = math.radians(_wrap_deg(float(yaw_deg)))
        c = math.cos(th)
        s = math.sin(th)
        # [dx_b; dy_b] = [c, s; -s, c] * [dx; dy]
        xb = c * float(x) + s * float(y)
        yb = -s * float(x) + c * float(y)
        dx_b.append(xb)
        dy_b.append(yb)
        dz_b.append(float(z))
        r_norm.append(math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2))

    print(f"csv: {args.csv}")
    print(f"rows_used: {len(dx_b)}")
    if "gnss_degraded" in df.columns and not args.include_degraded:
        print("filter: gnss_degraded==0 only")
    print("")
    print(_stats("body_dx (m)", dx_b))
    print(_stats("body_dy (m)", dy_b))
    print(_stats("body_dz (m)", dz_b))
    print(_stats("res_norm (m)", r_norm))
    print("")
    print(
        "Suggested lever arm (LiDAR/base -> GNSS antenna), use MEDIAN as a robust estimate:\n"
        f"  [{_quantile(dx_b, 0.5):.3f}, {_quantile(dy_b, 0.5):.3f}, {_quantile(dz_b, 0.5):.3f}]  (meters)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

