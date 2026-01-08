#!/usr/bin/env python3
"""
从 /fixposition/fpa/odomenu 构建 Fixposition(FP_*) 的最小 TF 子树，并把它与 LIO-SAM 的 TF 树联通。

为什么需要它：
  - 你的 bag/回放流程可能不包含 /tf(/tf_static)，因此 TF 树里没有 FP_ENU0/FP_POI/FP_VRTK。
  - 但 /fixposition/fpa/odomenu 明确提供了 “FP_POI 在 FP_ENU0 中的位姿”，足够重建关键 TF。

本脚本做三件事（尽量少但够用）：
  1) 发布动态 TF:  FP_ENU0 -> FP_POI        (来自 /fixposition/fpa/odomenu.pose.pose)
  2) 发布静态 TF:  FP_POI  -> FP_VRTK       (默认 identity；rawimu 的 frame_id 用到)
  3) 发布静态 TF:  FP_ENU0 -> map           (把 RTK ENU 树与机器人树联通，且与 params 一致)

其中 (3) 与 LIO-SAM 的 gpsExtrinsicRot 定义保持一致：
  - LIO-SAM: gps_lidar = gpsExtrinsicRot * gps_enu   (ENU -> map/lidar 的坐标轴旋转)
  - TF:      FP_ENU0 -> map  的旋转使用 R_enu_map = (gpsExtrinsicRot)^T
  - 平移使用第一帧 p0 (FP_POI 在 FP_ENU0 的位置)，与 zero_initial_position=true 对齐。

运行：
  source /root/autodl-tmp/catkin_ws/devel/setup.bash
  python3 scripts/bridge_fpa_enu_to_map_tf.py
"""

from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from fixposition_driver_msgs.msg import FpaOdomenu
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler, quaternion_from_matrix


def _normalize_frame(s: str) -> str:
    return str(s).strip().lstrip("/")


def _mat3_from_row_major(v: List[float]) -> np.ndarray:
    if not isinstance(v, list) or len(v) != 9:
        return np.eye(3, dtype=float)
    return np.asarray([float(x) for x in v], dtype=float).reshape(3, 3)


def _quat_xyzw_from_R(R: np.ndarray):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    qx, qy, qz, qw = quaternion_from_matrix(T)  # (x,y,z,w)
    return float(qx), float(qy), float(qz), float(qw)


@dataclass
class _State:
    enu_frame: Optional[str] = None
    poi_frame: Optional[str] = None
    published_enu_to_map: bool = False
    published_poi_to_vrtk: bool = False


class FpaTfBridge:
    def __init__(self) -> None:
        self.odomenu_topic = rospy.get_param("~odomenu_topic", "/fixposition/fpa/odomenu")
        self.wait_timeout = float(rospy.get_param("~wait_timeout", 30.0))
        self.use_receive_time = bool(rospy.get_param("~use_receive_time", True))

        # Map frame defaults to LIO-SAM config so TF matches the running pipeline.
        self.map_frame = _normalize_frame(rospy.get_param("~map_frame", rospy.get_param("lio_sam/mapFrame", "map")))

        self.publish_odomenu_tf = bool(rospy.get_param("~publish_odomenu_tf", True))
        self.publish_poi_to_vrtk_tf = bool(rospy.get_param("~publish_poi_to_vrtk_tf", True))
        self.publish_enu_to_map_tf = bool(rospy.get_param("~publish_enu_to_map_tf", True))

        self.vrtk_frame = _normalize_frame(rospy.get_param("~vrtk_frame", "FP_VRTK"))
        # If set, overrides the child frame name used for the dynamic ENU->POI pose TF.
        # This is useful to avoid TF conflicts when you want to use "FP_POI" as a physical/static
        # mounting frame in the robot TF tree. Example: set to "FP_POI_GNSS".
        self.pose_frame_override = _normalize_frame(rospy.get_param("~pose_frame_override", "")) or None
        # If set, overrides the parent frame used for publishing the static POI->VRTK TF.
        # Default: use the same POI frame as the dynamic pose TF.
        self.poi_frame_for_vrtk = _normalize_frame(rospy.get_param("~poi_frame_for_vrtk", "")) or None

        self.poi_to_vrtk_xyz = rospy.get_param("~poi_to_vrtk_xyz", [0.0, 0.0, 0.0])
        self.poi_to_vrtk_rpy_deg = rospy.get_param("~poi_to_vrtk_rpy_deg", [0.0, 0.0, 0.0])

        # gpsExtrinsicRot: ENU -> map rotation used by LIO-SAM (mapOptimization).
        self.R_map_enu = _mat3_from_row_major(rospy.get_param("lio_sam/gpsExtrinsicRot", []))
        self.R_enu_map = self.R_map_enu.T

        self.dyn_br = tf2_ros.TransformBroadcaster()
        self.static_br = tf2_ros.StaticTransformBroadcaster()

        self.state = _State()
        self._last_enu_to_poi_stamp = None

        rospy.Subscriber(self.odomenu_topic, FpaOdomenu, self._cb, queue_size=200)

    def _wait_for_valid_time(self, timeout_sec: float = 5.0) -> bool:
        """
        Under /use_sim_time, rospy.Time.now() stays 0 until /clock starts.
        Some tools treat /tf_static timestamps as time-bound; avoid publishing static TF with stamp=0.
        """
        if rospy.Time.now().to_sec() > 0.0:
            return True
        deadline = time.monotonic() + float(timeout_sec)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if rospy.Time.now().to_sec() > 0.0:
                return True
            time.sleep(0.01)
        return rospy.Time.now().to_sec() > 0.0

    def _publish_static(self, parent: str, child: str, xyz, q_xyzw) -> None:
        tf_msg = TransformStamped()
        stamp = rospy.Time.now()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent
        tf_msg.child_frame_id = child
        tf_msg.transform.translation.x = float(xyz[0])
        tf_msg.transform.translation.y = float(xyz[1])
        tf_msg.transform.translation.z = float(xyz[2])
        tf_msg.transform.rotation.x = float(q_xyzw[0])
        tf_msg.transform.rotation.y = float(q_xyzw[1])
        tf_msg.transform.rotation.z = float(q_xyzw[2])
        tf_msg.transform.rotation.w = float(q_xyzw[3])
        self.static_br.sendTransform(tf_msg)

        if stamp.to_sec() == 0.0:
            # Publish once immediately (so TF exists for nodes that start early), then republish once
            # /clock becomes valid so tools that treat /tf_static stamps as time-bound (e.g. Foxglove)
            # don't consider the transform "too old".
            rospy.logwarn("Published static TF %s->%s with stamp=0; will republish once /clock is valid", parent, child)
            threading.Thread(
                target=self._republish_static_when_time_valid,
                args=(parent, child, tuple(float(x) for x in xyz), tuple(float(x) for x in q_xyzw)),
                daemon=True,
            ).start()

    def _republish_static_when_time_valid(self, parent: str, child: str, xyz: Tuple[float, float, float], q_xyzw) -> None:
        if not self._wait_for_valid_time(timeout_sec=10.0):
            return

        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = parent
        tf_msg.child_frame_id = child
        tf_msg.transform.translation.x = float(xyz[0])
        tf_msg.transform.translation.y = float(xyz[1])
        tf_msg.transform.translation.z = float(xyz[2])
        tf_msg.transform.rotation.x = float(q_xyzw[0])
        tf_msg.transform.rotation.y = float(q_xyzw[1])
        tf_msg.transform.rotation.z = float(q_xyzw[2])
        tf_msg.transform.rotation.w = float(q_xyzw[3])
        self.static_br.sendTransform(tf_msg)
        rospy.loginfo(
            "Republished static TF %s->%s at t=%.3f", parent, child, float(tf_msg.header.stamp.to_sec())
        )

    def _cb(self, msg: FpaOdomenu) -> None:
        enu_frame = _normalize_frame(msg.header.frame_id) or "FP_ENU0"
        poi_frame_msg = _normalize_frame(getattr(msg, "pose_frame", "")) or "FP_POI"
        poi_frame = self.pose_frame_override or poi_frame_msg

        if self.publish_odomenu_tf:
            tf_msg = TransformStamped()
            stamp = rospy.Time.now() if self.use_receive_time else msg.header.stamp
            # With /use_sim_time + rosbag playback it's possible to receive multiple odomenu messages at
            # the exact same /clock time. Publishing TF repeatedly with identical timestamps triggers
            # TF_REPEATED_DATA warnings and is ignored by tf2 anyway. Drop exact duplicates.
            if self._last_enu_to_poi_stamp is not None:
                if stamp < self._last_enu_to_poi_stamp:
                    # Time jumped backwards (e.g. bag restarted) - reset the duplicate filter.
                    self._last_enu_to_poi_stamp = None
                elif stamp == self._last_enu_to_poi_stamp:
                    return

            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = enu_frame
            tf_msg.child_frame_id = poi_frame
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            tf_msg.transform.translation.x = float(p.x)
            tf_msg.transform.translation.y = float(p.y)
            tf_msg.transform.translation.z = float(p.z)
            tf_msg.transform.rotation.x = float(q.x)
            tf_msg.transform.rotation.y = float(q.y)
            tf_msg.transform.rotation.z = float(q.z)
            tf_msg.transform.rotation.w = float(q.w)
            self.dyn_br.sendTransform(tf_msg)
            self._last_enu_to_poi_stamp = stamp

        # Publish static TFs once (after we know the real ENU/POI frame names).
        if self.publish_poi_to_vrtk_tf and not self.state.published_poi_to_vrtk:
            poi_parent_for_vrtk = self.poi_frame_for_vrtk or poi_frame
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if isinstance(self.poi_to_vrtk_xyz, list) and len(self.poi_to_vrtk_xyz) == 3:
                xyz = [float(x) for x in self.poi_to_vrtk_xyz]
            else:
                rospy.logwarn("~poi_to_vrtk_xyz invalid, using [0,0,0]")
            if isinstance(self.poi_to_vrtk_rpy_deg, list) and len(self.poi_to_vrtk_rpy_deg) == 3:
                rpy = [float(x) for x in self.poi_to_vrtk_rpy_deg]
            else:
                rospy.logwarn("~poi_to_vrtk_rpy_deg invalid, using [0,0,0]")

            qx, qy, qz, qw = quaternion_from_euler(
                math.radians(rpy[0]),
                math.radians(rpy[1]),
                math.radians(rpy[2]),
            )
            self._publish_static(poi_parent_for_vrtk, self.vrtk_frame, xyz, (qx, qy, qz, qw))
            self.state.published_poi_to_vrtk = True
            rospy.loginfo(
                "Published static TF: %s -> %s (xyz=%s rpy_deg=%s)", poi_parent_for_vrtk, self.vrtk_frame, xyz, rpy
            )

        if self.publish_enu_to_map_tf and not self.state.published_enu_to_map:
            p0 = msg.pose.pose.position
            t = (float(p0.x), float(p0.y), float(p0.z))
            qx, qy, qz, qw = _quat_xyzw_from_R(self.R_enu_map)
            self._publish_static(enu_frame, self.map_frame, t, (qx, qy, qz, qw))

            yaw_deg = math.degrees(math.atan2(float(self.R_map_enu[1, 0]), float(self.R_map_enu[0, 0])))
            self.state.published_enu_to_map = True
            self.state.enu_frame = enu_frame
            self.state.poi_frame = poi_frame
            rospy.loginfo(
                "Published static TF: %s -> %s, t0=(%.3f,%.3f,%.3f), gpsExtrinsicRot_yaw(deg)=%.2f",
                enu_frame,
                self.map_frame,
                t[0],
                t[1],
                t[2],
                yaw_deg,
            )

    def wait_ready(self) -> None:
        deadline = time.monotonic() + float(self.wait_timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if (not self.publish_enu_to_map_tf or self.state.published_enu_to_map) and (
                not self.publish_poi_to_vrtk_tf or self.state.published_poi_to_vrtk
            ):
                return
            time.sleep(0.05)
        if self.publish_enu_to_map_tf and not self.state.published_enu_to_map:
            rospy.logwarn("Did not receive odomenu within %.1fs; FP_ENU0->map not published yet", self.wait_timeout)


def main() -> None:
    rospy.init_node("bridge_fpa_enu_to_map_tf", anonymous=True)
    node = FpaTfBridge()
    rospy.loginfo("bridge_fpa_enu_to_map_tf: subscribing %s", node.odomenu_topic)
    node.wait_ready()
    rospy.spin()


if __name__ == "__main__":
    main()
