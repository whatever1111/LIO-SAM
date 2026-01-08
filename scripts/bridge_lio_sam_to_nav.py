#!/usr/bin/env python3
"""
Bridge LIO-SAM odometry TF tree to legacy navigation interfaces.

Publishes:
  - TF: map -> lio_map (configurable; set from /initialpose)
  - Topic: /odom (nav_msgs/Odometry) in map frame
  - Topic: /location_status (std_msgs/Int8) as a simple health gate

Use case:
  Existing system expects:
    - map is the global frame (map_server + move_base)
    - /odom is in map frame @ ~10Hz
    - /location_status indicates localization OK/LOST
  LIO-SAM provides:
    - lio_map->lio_odom TF (identity) + lio_odom->base_link TF (local)
  This node adds:
    - map->lio_map alignment (set once via /initialpose, or keep identity if not used)
    - /odom republishing compatible with the legacy stack
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import tf.transformations as tft

from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8


def _normalize_frame(frame_id: str) -> str:
    return (frame_id or "").strip().lstrip("/")


def _matrix_from_translation_quaternion(
    translation_xyz: Tuple[float, float, float],
    quaternion_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    m = tft.quaternion_matrix(quaternion_xyzw)
    m[0:3, 3] = np.array(translation_xyz, dtype=float)
    return m


def _translation_quaternion_from_matrix(m: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    t = (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
    q = tuple(float(v) for v in tft.quaternion_from_matrix(m))
    return t, q


def _yaw_from_quaternion(q_xyzw: Tuple[float, float, float, float]) -> float:
    r, p, y = tft.euler_from_quaternion(q_xyzw)
    return float(y)


def _wrap_to_pi(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _transform_to_matrix(tr: TransformStamped) -> np.ndarray:
    t = tr.transform.translation
    r = tr.transform.rotation
    return _matrix_from_translation_quaternion((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))


@dataclass
class _PoseSample:
    stamp: rospy.Time
    mat_map_base: np.ndarray  # 4x4


class LioSamNavBridge:
    def __init__(self) -> None:
        self.map_frame = _normalize_frame(rospy.get_param("~map_frame", "map"))
        self.lio_map_frame = _normalize_frame(rospy.get_param("~lio_map_frame", "lio_map"))
        self.base_frame = _normalize_frame(rospy.get_param("~base_frame", "base_link"))

        self.publish_rate_hz = float(rospy.get_param("~publish_rate", 10.0))
        self.stale_timeout_s = float(rospy.get_param("~stale_timeout", 0.5))

        self.publish_map_to_lio_map = bool(rospy.get_param("~publish_map_to_lio_map", True))
        self.use_initialpose = bool(rospy.get_param("~use_initialpose", True))

        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.child_frame_id = str(rospy.get_param("~child_frame_id", ""))  # legacy stack uses empty string
        self.zero_angular_twist = bool(rospy.get_param("~zero_angular_twist", True))

        self.location_status_topic = str(rospy.get_param("~location_status_topic", "/location_status"))
        self.location_status_ok = int(rospy.get_param("~location_status_ok", 1))
        self.location_status_lost = int(rospy.get_param("~location_status_lost", 0))

        self.initial_map_to_lio_map = rospy.get_param("~initial_map_to_lio_map", None)
        # Backwards-compatible alias (older versions used map->odom here).
        if self.initial_map_to_lio_map is None:
            self.initial_map_to_lio_map = rospy.get_param("~initial_map_to_odom", None)

        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()

        self._pub_odom = rospy.Publisher(self.odom_topic, Odometry, queue_size=10)
        self._pub_location_status = rospy.Publisher(self.location_status_topic, Int8, queue_size=10)

        self._sub_initialpose: Optional[rospy.Subscriber] = None
        self._pending_initialpose: Optional[PoseWithCovarianceStamped] = None
        if self.use_initialpose:
            self._sub_initialpose = rospy.Subscriber(
                "/initialpose", PoseWithCovarianceStamped, self._on_initialpose, queue_size=1
            )

        # map_T_lio_map: default identity unless set by param or /initialpose
        self._map_T_lio_map = np.eye(4, dtype=float)
        if isinstance(self.initial_map_to_lio_map, (list, tuple)) and len(self.initial_map_to_lio_map) >= 6:
            x, y, z, roll, pitch, yaw = (float(v) for v in self.initial_map_to_lio_map[:6])
            q = tft.quaternion_from_euler(roll, pitch, yaw)
            self._map_T_lio_map = _matrix_from_translation_quaternion((x, y, z), q)
            rospy.loginfo("bridge_lio_sam_to_nav: init map->lio_map from param: xyz=(%.3f,%.3f,%.3f) rpy=(%.3f,%.3f,%.3f)",
                          x, y, z, roll, pitch, yaw)

        self._prev_sample: Optional[_PoseSample] = None
        self._last_good_stamp: Optional[rospy.Time] = None

        period = 1.0 / max(self.publish_rate_hz, 1e-3)
        self._timer = rospy.Timer(rospy.Duration(period), self._on_timer)

        rospy.loginfo(
            "bridge_lio_sam_to_nav: map=%s lio_map=%s base=%s out_odom=%s out_status=%s rate=%.2fHz",
            self.map_frame,
            self.lio_map_frame,
            self.base_frame,
            self.odom_topic,
            self.location_status_topic,
            self.publish_rate_hz,
        )

    def _on_initialpose(self, msg: PoseWithCovarianceStamped) -> None:
        # Store; alignment will be attempted on next timer tick (tf may not be ready in callback).
        self._pending_initialpose = msg
        rospy.logwarn_throttle(1.0, "bridge_lio_sam_to_nav: received /initialpose, pending alignment update")

    def _try_update_alignment_from_initialpose(self, now: rospy.Time) -> None:
        if self._pending_initialpose is None:
            return

        msg = self._pending_initialpose
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else now

        try:
            tr_lio_map_base = self._tf_buffer.lookup_transform(
                self.lio_map_frame, self.base_frame, stamp, rospy.Duration(0.2)
            )
        except Exception as ex:
            rospy.logwarn_throttle(1.0, "bridge_lio_sam_to_nav: cannot lookup %s->%s for alignment: %s",
                                   self.lio_map_frame, self.base_frame, str(ex))
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        map_T_base = _matrix_from_translation_quaternion((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
        lio_map_T_base = _transform_to_matrix(tr_lio_map_base)

        # map_T_lio_map = map_T_base * inv(lio_map_T_base)
        try:
            self._map_T_lio_map = map_T_base @ np.linalg.inv(lio_map_T_base)
        except Exception as ex:
            rospy.logwarn("bridge_lio_sam_to_nav: failed to compute alignment: %s", str(ex))
            return

        t, quat = _translation_quaternion_from_matrix(self._map_T_lio_map)
        rospy.loginfo(
            "bridge_lio_sam_to_nav: updated map->lio_map from /initialpose: t=(%.3f,%.3f,%.3f) yaw=%.1fdeg",
            t[0],
            t[1],
            t[2],
            _yaw_from_quaternion(quat) * 180.0 / math.pi,
        )
        self._pending_initialpose = None

    def _broadcast_map_to_lio_map(self, stamp: rospy.Time) -> None:
        if not self.publish_map_to_lio_map:
            return

        t, q = _translation_quaternion_from_matrix(self._map_T_lio_map)
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.lio_map_frame
        msg.transform.translation.x = t[0]
        msg.transform.translation.y = t[1]
        msg.transform.translation.z = t[2]
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]
        self._tf_broadcaster.sendTransform(msg)

    def _compute_twist(self, now_sample: _PoseSample) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if self._prev_sample is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        dt = (now_sample.stamp - self._prev_sample.stamp).to_sec()
        if dt <= 1e-4 or dt > 1.0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        p_now = now_sample.mat_map_base[0:3, 3]
        p_prev = self._prev_sample.mat_map_base[0:3, 3]
        v_map = (p_now - p_prev) / float(dt)

        # Express linear velocity in base_link frame (like typical odom consumer expectation).
        R_map_base = now_sample.mat_map_base[0:3, 0:3]
        v_base = R_map_base.T @ v_map

        # Yaw rate from quaternion delta (assume roll/pitch small; keep only yaw).
        _, q_now = _translation_quaternion_from_matrix(now_sample.mat_map_base)
        _, q_prev = _translation_quaternion_from_matrix(self._prev_sample.mat_map_base)
        yaw_now = _yaw_from_quaternion(q_now)
        yaw_prev = _yaw_from_quaternion(q_prev)
        yaw_rate = _wrap_to_pi(yaw_now - yaw_prev) / float(dt)

        if self.zero_angular_twist:
            yaw_rate = 0.0

        linear = (float(v_base[0]), float(v_base[1]), float(v_base[2]))
        angular = (0.0, 0.0, float(yaw_rate))
        return linear, angular

    def _publish_location_status(self, stamp: rospy.Time, ok: bool) -> None:
        msg = Int8()
        msg.data = int(self.location_status_ok if ok else self.location_status_lost)
        self._pub_location_status.publish(msg)

    def _publish_odom(self, stamp: rospy.Time, map_T_base: np.ndarray) -> None:
        (tx, ty, tz), (qx, qy, qz, qw) = _translation_quaternion_from_matrix(map_T_base)

        now_sample = _PoseSample(stamp=stamp, mat_map_base=map_T_base)
        linear, angular = self._compute_twist(now_sample)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = tx
        odom.pose.pose.position.y = ty
        odom.pose.pose.position.z = tz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = linear[0]
        odom.twist.twist.linear.y = linear[1]
        odom.twist.twist.linear.z = linear[2]
        odom.twist.twist.angular.x = angular[0]
        odom.twist.twist.angular.y = angular[1]
        odom.twist.twist.angular.z = angular[2]

        # Keep covariances all-zero for legacy compatibility.
        self._pub_odom.publish(odom)
        self._prev_sample = now_sample

    def _on_timer(self, event: rospy.TimerEvent) -> None:
        now = rospy.Time.now()

        # Try apply /initialpose alignment if any.
        self._try_update_alignment_from_initialpose(now)

        # Always broadcast map->lio_map (even if lio_map->base is missing) so TF tree is stable.
        self._broadcast_map_to_lio_map(now)

        try:
            tr_lio_map_base = self._tf_buffer.lookup_transform(
                self.lio_map_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.1)
            )
        except Exception as ex:
            self._publish_location_status(now, ok=False)
            rospy.logwarn_throttle(1.0, "bridge_lio_sam_to_nav: missing TF %s->%s: %s",
                                   self.lio_map_frame, self.base_frame, str(ex))
            return

        lio_map_T_base = _transform_to_matrix(tr_lio_map_base)
        map_T_base = self._map_T_lio_map @ lio_map_T_base

        pose_stamp = tr_lio_map_base.header.stamp if tr_lio_map_base.header.stamp != rospy.Time(0) else now
        self._publish_odom(pose_stamp, map_T_base)

        # Health: TF is considered "fresh" if its stamp is recent (or if stamp is unset).
        ok = True
        tf_stamp = tr_lio_map_base.header.stamp
        if tf_stamp and tf_stamp != rospy.Time(0):
            age = (now - tf_stamp).to_sec()
            ok = age <= self.stale_timeout_s
        self._last_good_stamp = now
        self._publish_location_status(now, ok=ok)


def main() -> None:
    rospy.init_node("bridge_lio_sam_to_nav", anonymous=False)
    _ = LioSamNavBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
