#!/usr/bin/env python3
"""
离线梳理 bag 中的 TF / 传感器 frame_id，并给出 LIO-SAM 所需 TF 关系建议。

用法:
  python3 scripts/report_bag_tf.py --bag <path_to_bag>
  python3 scripts/report_bag_tf.py --bag <path_to_bag> --params config/params.yaml
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


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


def _import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"无法 import yaml(PyYAML): {exc}")


@dataclass(frozen=True)
class Transform:
    parent: str
    child: str
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float

    def is_identity(self, trans_tol: float = 1e-6, rot_tol_rad: float = 1e-6) -> bool:
        t_norm = math.sqrt(self.tx * self.tx + self.ty * self.ty + self.tz * self.tz)
        if t_norm > trans_tol:
            return False
        # quaternion angle to identity: angle = 2*acos(|w|)
        w = max(-1.0, min(1.0, abs(self.qw)))
        angle = 2.0 * math.acos(w)
        return angle <= rot_tol_rad


def _normalize_frame(frame_id: str) -> str:
    return frame_id.strip().lstrip("/")


def _first_header_frame_id(bag, topic: str) -> Optional[str]:
    for _, msg, _ in bag.read_messages(topics=[topic]):
        hdr = getattr(msg, "header", None)
        if hdr is None:
            return None
        return _normalize_frame(getattr(hdr, "frame_id", ""))
    return None


def _find_tf(bag, parent: str, child: str, topics=("/tf_static", "/tf")) -> Optional[Transform]:
    parent = _normalize_frame(parent)
    child = _normalize_frame(child)
    for topic, msg, _ in bag.read_messages(topics=list(topics)):
        for tr in msg.transforms:
            p = _normalize_frame(tr.header.frame_id)
            c = _normalize_frame(tr.child_frame_id)
            if p == parent and c == child:
                t = tr.transform.translation
                r = tr.transform.rotation
                return Transform(
                    parent=p,
                    child=c,
                    tx=float(t.x),
                    ty=float(t.y),
                    tz=float(t.z),
                    qx=float(r.x),
                    qy=float(r.y),
                    qz=float(r.z),
                    qw=float(r.w),
                )
    return None


def _load_lio_sam_frames(params_path: Path) -> Tuple[str, str, str, str]:
    yaml = _import_yaml()
    data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    lio = (data or {}).get("lio_sam", {})
    lidar = _normalize_frame(str(lio.get("lidarFrame", "lidar_link")))
    base = _normalize_frame(str(lio.get("baselinkFrame", "base_link")))
    odom = _normalize_frame(str(lio.get("odometryFrame", "odom")))
    mapf = _normalize_frame(str(lio.get("mapFrame", "map")))
    return lidar, base, odom, mapf


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_params = repo_root / "config" / "params.yaml"

    ap = argparse.ArgumentParser(description="Report TF requirements from a ROS bag (offline).")
    ap.add_argument("--bag", required=True, help="ROS bag 路径")
    ap.add_argument("--params", default=str(default_params), help="LIO-SAM params.yaml 路径")
    ap.add_argument("--trans-tol", type=float, default=1e-6, help="判断 identity 的平移容差(m)")
    ap.add_argument("--rot-tol-deg", type=float, default=1e-3, help="判断 identity 的旋转容差(度)")
    args = ap.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    params_path = Path(args.params).expanduser().resolve()
    if not bag_path.is_file():
        raise SystemExit(f"bag 不存在: {bag_path}")
    if not params_path.is_file():
        raise SystemExit(f"params.yaml 不存在: {params_path}")

    rosbag = _import_rosbag()

    lidar_frame, base_frame, odom_frame, map_frame = _load_lio_sam_frames(params_path)

    print("=" * 80)
    print("TF / frame_id 离线梳理")
    print("=" * 80)
    print(f"bag: {bag_path}")
    print(f"params: {params_path}")
    print("")

    with rosbag.Bag(str(bag_path), "r") as bag:
        lidar_msg_frame = _first_header_frame_id(bag, "/lidar_points")
        imu_msg_frame = _first_header_frame_id(bag, "/imu/data")

        print("[bag 传感器 frame_id]")
        print(f"  /lidar_points: {lidar_msg_frame!r}")
        print(f"  /imu/data:     {imu_msg_frame!r}")
        print("")

        print("[LIO-SAM 参数帧]")
        print(f"  mapFrame:      {map_frame}")
        print(f"  odometryFrame: {odom_frame}")
        print(f"  baselinkFrame: {base_frame}")
        print(f"  lidarFrame:    {lidar_frame}")
        print("")

        print("[关键静态外参检查]")
        tf_bl = _find_tf(bag, base_frame, "lidar_link", topics=("/tf_static", "/tf"))
        if tf_bl is None:
            print(f"  - bag 中未找到 {base_frame} -> lidar_link")
        else:
            rot_tol_rad = float(args.rot_tol_deg) * math.pi / 180.0
            ok = tf_bl.is_identity(trans_tol=float(args.trans_tol), rot_tol_rad=rot_tol_rad)
            print(f"  - {tf_bl.parent} -> {tf_bl.child}: "
                  f"t=({tf_bl.tx:.6f},{tf_bl.ty:.6f},{tf_bl.tz:.6f}) "
                  f"q=({tf_bl.qx:.6f},{tf_bl.qy:.6f},{tf_bl.qz:.6f},{tf_bl.qw:.6f}) "
                  f"{'IDENTITY' if ok else 'NON-IDENTITY'}")
        print("")

    print("[运行时所需 TF 关系 (建议最小闭环)]")
    print(f"  {map_frame} -> {odom_frame}         (由 LIO-SAM TransformFusion 发布，通常为 identity)")
    print(f"  {odom_frame} -> {base_frame}        (由 LIO-SAM TransformFusion 发布)")
    print(f"  {base_frame} -> lidar_link          (已知与 base_link 重合时为 identity；可用 static_transform_publisher 发布)")
    print("")

    print("[建议的 static_transform_publisher]")
    print(f"  rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 {base_frame} lidar_link")
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

