#!/usr/bin/env python3
"""
Offline GNSS time-sync window analysis.

Reports dt = t_gps - t_odom under different association policies:
  - nearest:       choose min |dt| (no window needed)
  - latest_before: choose max t_gps <= t_odom within window (causal), else earliest-in-window
  - first:         choose earliest-in-window (legacy "queue front" behavior)

This helps you pick a reasonable gpsTimeWindow and validate that the chosen policy is not
systematically selecting stale measurements.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    if p <= 0:
        return v[0]
    if p >= 100:
        return v[-1]
    k = (len(v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return v[int(k)]
    return v[f] * (c - k) + v[c] * (k - f)


def _stats_ms(values: Sequence[float]) -> str:
    if not values:
        return "n=0"
    mn = min(values) * 1e3
    mx = max(values) * 1e3
    med = _percentile(values, 50) * 1e3
    p90 = _percentile(values, 90) * 1e3
    p99 = _percentile(values, 99) * 1e3
    return f"n={len(values)} median={med:.3f}ms p90={p90:.3f}ms p99={p99:.3f}ms min={mn:.3f}ms max={mx:.3f}ms"


def _load_topic_times(bag_path: str, topic: str) -> List[float]:
    try:
        import rosbag  # type: ignore
    except Exception as e:
        raise RuntimeError("rosbag python module not available; run inside a ROS environment") from e

    times: List[float] = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, _, t in bag.read_messages(topics=[topic]):
            times.append(t.to_sec())
    return times


@dataclass
class DegradedTimeline:
    t: List[float]
    v: List[int]

    def at(self, ts: float) -> int:
        if not self.t:
            return 0
        idx = bisect.bisect_right(self.t, ts) - 1
        if idx < 0:
            return 0
        return self.v[idx]


def _load_degraded_timeline(bag_path: str, topic: str) -> DegradedTimeline:
    try:
        import rosbag  # type: ignore
    except Exception as e:
        raise RuntimeError("rosbag python module not available; run inside a ROS environment") from e

    t_list: List[float] = []
    v_list: List[int] = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, t in bag.read_messages(topics=[topic]):
            t_list.append(t.to_sec())
            v_list.append(int(getattr(msg, "data", 0)))
    return DegradedTimeline(t=t_list, v=v_list)


def _dt_nearest(odom_t: Sequence[float], gps_t: Sequence[float]) -> List[float]:
    dt: List[float] = []
    j = 0
    for t in odom_t:
        while j < len(gps_t) and gps_t[j] < t:
            j += 1
        best: Optional[float] = None
        if j > 0:
            best = gps_t[j - 1]
        if j < len(gps_t):
            if best is None or abs(gps_t[j] - t) < abs(best - t):
                best = gps_t[j]
        if best is not None:
            dt.append(best - t)
    return dt


def _dt_latest_before(odom_t: Sequence[float], gps_t: Sequence[float], win: float) -> List[float]:
    dt: List[float] = []
    for t in odom_t:
        idxL = bisect.bisect_left(gps_t, t - win)
        idxR = bisect.bisect_right(gps_t, t + win)
        if idxL >= idxR:
            continue
        idxBefore = bisect.bisect_right(gps_t, t) - 1
        if idxBefore < idxL:
            chosen = gps_t[idxL]  # earliest-in-window (future)
        else:
            chosen = gps_t[idxBefore]  # latest <= t
        dt.append(chosen - t)
    return dt


def _dt_first(odom_t: Sequence[float], gps_t: Sequence[float], win: float) -> List[float]:
    dt: List[float] = []
    for t in odom_t:
        idxL = bisect.bisect_left(gps_t, t - win)
        idxR = bisect.bisect_right(gps_t, t + win)
        if idxL >= idxR:
            continue
        dt.append(gps_t[idxL] - t)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Input rosbag path")
    ap.add_argument("--odom-topic", default="/lio_sam/mapping/odometry", help="Odometry/keyframe time base topic")
    ap.add_argument("--gps-topic", default="/odometry/gps", help="GNSS odometry topic to associate")
    ap.add_argument("--degraded-topic", default="/gnss_degraded", help="Optional Bool topic marking degraded GNSS")
    ap.add_argument(
        "--windows",
        type=float,
        nargs="*",
        default=[0.05, 0.1, 0.2],
        help="Time windows (seconds) to evaluate for latest_before/first",
    )
    ap.add_argument("--no-degraded", action="store_true", help="Skip degraded breakdown even if topic exists")
    args = ap.parse_args()

    odom_t = _load_topic_times(args.bag, args.odom_topic)
    gps_t = _load_topic_times(args.bag, args.gps_topic)
    if not odom_t:
        print(f"[ERR] No messages on odom topic: {args.odom_topic}", file=sys.stderr)
        return 2
    if not gps_t:
        print(f"[ERR] No messages on gps topic: {args.gps_topic}", file=sys.stderr)
        return 2

    odom_t.sort()
    gps_t.sort()

    print(f"bag: {args.bag}")
    print(f"odom_topic: {args.odom_topic} ({len(odom_t)} msgs)")
    print(f"gps_topic:  {args.gps_topic} ({len(gps_t)} msgs)")
    print("")

    dt_nearest = _dt_nearest(odom_t, gps_t)
    print("[nearest]        ", _stats_ms(dt_nearest))

    for win in args.windows:
        if win <= 0:
            continue
        dt_lb = _dt_latest_before(odom_t, gps_t, win)
        dt_first = _dt_first(odom_t, gps_t, win)
        print(f"[latest_before]  win={win:.3f}s  {_stats_ms(dt_lb)}")
        print(f"[first]          win={win:.3f}s  {_stats_ms(dt_first)}")

    # Degraded breakdown (if available)
    if not args.no_degraded:
        try:
            timeline = _load_degraded_timeline(args.bag, args.degraded_topic)
            if timeline.t:
                dt0: List[float] = []
                dt1: List[float] = []
                for t, dt in zip(odom_t[: len(dt_nearest)], dt_nearest):
                    (dt1 if timeline.at(t) else dt0).append(dt)
                print("")
                print("[nearest by degraded]")
                print("  degraded=0", _stats_ms(dt0))
                print("  degraded=1", _stats_ms(dt1))
        except Exception:
            # Silently ignore if topic missing or not readable.
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

