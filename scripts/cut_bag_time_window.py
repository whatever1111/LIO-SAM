#!/usr/bin/env python3
"""
Cut a time window from a ROS bag, while keeping critical latched topics usable.

Why this exists:
  - Many bags contain /tf_static and /map only once at the beginning.
  - If you cut from t=60s, these one-time messages are lost and TF tree becomes incomplete.
  - This script injects a single /tf_static (and optionally /map) message at the window start time.

Example:
  source /opt/ros/noetic/setup.bash
  python3 scripts/cut_bag_time_window.py \
    --in  /root/autodl-tmp/st_0106_no_loc.bag \
    --out /root/autodl-tmp/st_0106_no_loc_60_300.bag \
    --start 60 --end 300 --inject-tf-static --inject-map
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import OrderedDict
from typing import Dict, Optional, Tuple


def _import_rosbag():
    try:
        import rosbag  # type: ignore

        return rosbag
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Cannot import rosbag. Did you source the ROS environment?\n"
            "  source /opt/ros/noetic/setup.bash\n"
            f"Original error: {exc}"
        )


def _normalize_frame(s: str) -> str:
    return (s or "").strip().lstrip("/")


def _copy_tf_static_message_at_time(
    inbag,
    outbag,
    out_time,
    *,
    set_transform_stamp_to_zero: bool,
) -> bool:
    """
    Build a single tf2_msgs/TFMessage containing all unique static transforms from the input bag
    and write it once to /tf_static at out_time.

    Returns True if written, False if /tf_static does not exist in input bag.
    """
    try:
        from tf2_msgs.msg import TFMessage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Cannot import tf2_msgs/TFMessage: {exc}")

    tf_static_header = None
    static_by_edge: Dict[Tuple[str, str], object] = OrderedDict()

    for topic, msg, _, conn_header in inbag.read_messages(
        topics=["/tf_static"], return_connection_header=True
    ):
        if tf_static_header is None and conn_header is not None:
            tf_static_header = dict(conn_header)
        for tr in msg.transforms:
            parent = _normalize_frame(getattr(tr.header, "frame_id", ""))
            child = _normalize_frame(getattr(tr, "child_frame_id", ""))
            static_by_edge[(parent, child)] = tr

    if not static_by_edge:
        return False

    # Deterministic order: sort by (parent, child)
    tf_msg = TFMessage()
    tf_msg.transforms = []
    for (parent, child) in sorted(static_by_edge.keys()):
        tr = copy.deepcopy(static_by_edge[(parent, child)])
        tr.header.frame_id = parent
        tr.child_frame_id = child
        if set_transform_stamp_to_zero:
            tr.header.stamp.secs = 0
            tr.header.stamp.nsecs = 0
        tf_msg.transforms.append(tr)

    header = tf_static_header or {
        "topic": "/tf_static",
        "type": tf_msg.__class__._type,
        "md5sum": tf_msg.__class__._md5sum,
        "message_definition": tf_msg._full_text,
        "latching": "1",
    }
    header["topic"] = "/tf_static"
    header["type"] = tf_msg.__class__._type
    header["md5sum"] = tf_msg.__class__._md5sum
    header["message_definition"] = tf_msg._full_text
    header.setdefault("latching", "1")

    outbag.write("/tf_static", tf_msg, t=out_time, connection_header=header)
    return True


def _copy_map_message_at_time(inbag, outbag, out_time) -> bool:
    """Copy the first /map message (if any) to the output bag at out_time."""
    map_header = None
    map_msg = None
    for topic, msg, _, conn_header in inbag.read_messages(
        topics=["/map"], return_connection_header=True
    ):
        map_msg = msg
        if conn_header is not None:
            map_header = dict(conn_header)
        break

    if map_msg is None:
        return False

    header = map_header or {
        "topic": "/map",
        "type": map_msg.__class__._type,
        "md5sum": map_msg.__class__._md5sum,
        "message_definition": map_msg._full_text,
        "latching": "1",
    }
    header["topic"] = "/map"
    header["type"] = map_msg.__class__._type
    header["md5sum"] = map_msg.__class__._md5sum
    header["message_definition"] = map_msg._full_text
    header.setdefault("latching", "1")

    outbag.write("/map", map_msg, t=out_time, connection_header=header)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Cut a time window from a ROS bag, injecting /tf_static (and /map).")
    ap.add_argument("--in", dest="in_bag", required=True, help="Input bag path")
    ap.add_argument("--out", dest="out_bag", required=True, help="Output bag path")
    ap.add_argument("--start", type=float, required=True, help="Window start offset (seconds from bag start)")
    ap.add_argument("--end", type=float, required=True, help="Window end offset (seconds from bag start)")
    ap.add_argument("--inject-tf-static", action="store_true", help="Inject one aggregated /tf_static at window start")
    ap.add_argument("--inject-map", action="store_true", help="Inject the first /map at window start")
    ap.add_argument(
        "--keep-original-tf-static",
        action="store_true",
        help="Also keep original /tf_static messages inside the window (default: drop and keep only injected)",
    )
    ap.add_argument(
        "--tf-static-stamp-zero",
        action="store_true",
        help="Set each TransformStamped.header.stamp in injected /tf_static to zero (recommended for tf2 static cache)",
    )
    ap.add_argument("--progress-every", type=int, default=20000, help="Print progress every N messages copied")
    args = ap.parse_args()

    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")

    rosbag = _import_rosbag()
    try:
        import genpy  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Cannot import genpy: {exc}")

    in_path = args.in_bag
    out_path = args.out_bag

    t0_wall = time.monotonic()
    with rosbag.Bag(in_path, "r") as inbag:
        bag_start = float(inbag.get_start_time())
        bag_end = float(inbag.get_end_time())
        t_start = bag_start + float(args.start)
        t_end = bag_start + float(args.end)

        if t_start < bag_start or t_start > bag_end:
            raise SystemExit(f"start offset out of range: t_start={t_start:.3f}, bag=[{bag_start:.3f},{bag_end:.3f}]")
        if t_end < bag_start or t_end > bag_end:
            raise SystemExit(f"end offset out of range: t_end={t_end:.3f}, bag=[{bag_start:.3f},{bag_end:.3f}]")

        gt_start = genpy.Time.from_sec(t_start)
        gt_end = genpy.Time.from_sec(t_end)

        print(f"[cut_bag] in={in_path}")
        print(f"[cut_bag] out={out_path}")
        print(f"[cut_bag] bag_start={bag_start:.3f} bag_end={bag_end:.3f} duration={bag_end - bag_start:.3f}s")
        print(f"[cut_bag] window: +{args.start:.3f}s..+{args.end:.3f}s  =>  [{t_start:.3f}..{t_end:.3f}]")

        with rosbag.Bag(out_path, "w") as outbag:
            # Inject latched topics at window start so they are available even when the original was only recorded at t=0.
            if args.inject_tf_static:
                wrote = _copy_tf_static_message_at_time(
                    inbag,
                    outbag,
                    gt_start,
                    set_transform_stamp_to_zero=bool(args.tf_static_stamp_zero),
                )
                print(f"[cut_bag] inject /tf_static: {'OK' if wrote else 'SKIP (no /tf_static in input)'}")

            if args.inject_map:
                wrote = _copy_map_message_at_time(inbag, outbag, gt_start)
                print(f"[cut_bag] inject /map: {'OK' if wrote else 'SKIP (no /map in input)'}")

            copied = 0
            dropped_tf_static = 0
            for topic, msg, t, conn_header in inbag.read_messages(
                start_time=gt_start, end_time=gt_end, return_connection_header=True
            ):
                if topic == "/tf_static" and args.inject_tf_static and not args.keep_original_tf_static:
                    dropped_tf_static += 1
                    continue

                outbag.write(topic, msg, t=t, connection_header=conn_header)
                copied += 1
                if args.progress_every > 0 and copied % int(args.progress_every) == 0:
                    dt = time.monotonic() - t0_wall
                    print(f"[cut_bag] copied={copied} dropped_tf_static={dropped_tf_static} elapsed={dt:.1f}s")

    dt = time.monotonic() - t0_wall
    print(f"[cut_bag] DONE in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

