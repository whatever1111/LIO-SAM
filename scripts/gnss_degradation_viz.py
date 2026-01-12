#!/usr/bin/env python3
"""
GNSS degradation visualization helper for RViz.

Publishes:
  - nav_msgs/Path:   degraded segment poses (so you can overlay in RViz)
  - visualization_msgs/Marker: a "GNSS DEGRADED" text flag near the current pose

Designed to be lightweight and not modify core LIO-SAM logic.
"""

import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker


class GnssDegradationViz:
    def __init__(self):
        self.odom_topic = str(rospy.get_param("~odom_topic", "/lio_sam/mapping/odometry"))
        self.degraded_topic = str(rospy.get_param("~degraded_topic", "/gnss_degraded"))
        self.path_topic = str(rospy.get_param("~path_topic", "/lio_sam/gnss/degraded_path"))
        self.marker_topic = str(rospy.get_param("~marker_topic", "/lio_sam/gnss/degraded_marker"))

        self.frame_id = str(rospy.get_param("~frame_id", ""))  # empty => follow odom header.frame_id
        self.max_path_len = int(rospy.get_param("~max_path_len", 20000))
        self.min_add_distance = float(rospy.get_param("~min_add_distance", 0.0))  # meters, 0=disable

        self.text_height = float(rospy.get_param("~text_height", 0.8))
        self.text_z_offset = float(rospy.get_param("~text_z_offset", 1.5))
        self.text_alpha = float(rospy.get_param("~text_alpha", 0.95))

        self._lock = threading.Lock()
        self._degraded = False
        self._last_pose_xy = None

        self._path = Path()

        self._path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1)
        self._marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)

        rospy.Subscriber(self.degraded_topic, Bool, self._degraded_cb, queue_size=200)
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=200)

        rospy.loginfo("GNSS degradation viz started")
        rospy.loginfo("  odom_topic:      %s", self.odom_topic)
        rospy.loginfo("  degraded_topic:  %s", self.degraded_topic)
        rospy.loginfo("  path_topic:      %s", self.path_topic)
        rospy.loginfo("  marker_topic:    %s", self.marker_topic)

    def _degraded_cb(self, msg):
        with self._lock:
            self._degraded = bool(msg.data)

    def _odom_cb(self, msg):
        with self._lock:
            degraded = self._degraded

        frame_id = self.frame_id or msg.header.frame_id or "map"

        # Publish a "flag" marker at current pose
        marker = Marker()
        marker.header.stamp = msg.header.stamp
        marker.header.frame_id = frame_id
        marker.ns = "gnss_degraded"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = msg.pose.pose.position.x
        marker.pose.position.y = msg.pose.pose.position.y
        marker.pose.position.z = msg.pose.pose.position.z + self.text_z_offset
        marker.pose.orientation.w = 1.0
        marker.scale.z = max(0.01, self.text_height)

        if degraded:
            marker.text = "GNSS DEGRADED"
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = max(0.0, min(1.0, self.text_alpha))
        else:
            # Hide marker when GNSS is good
            marker.text = ""
            marker.color.a = 0.0

        self._marker_pub.publish(marker)

        # Append degraded poses to a Path for RViz overlay
        if not degraded:
            return

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = frame_id
        pose.pose = msg.pose.pose

        with self._lock:
            # Optional distance-based downsampling
            if self.min_add_distance > 0.0:
                if self._last_pose_xy is not None:
                    dx = float(pose.pose.position.x) - float(self._last_pose_xy[0])
                    dy = float(pose.pose.position.y) - float(self._last_pose_xy[1])
                    if math.hypot(dx, dy) < self.min_add_distance:
                        return
                self._last_pose_xy = (float(pose.pose.position.x), float(pose.pose.position.y))

            self._path.header.stamp = msg.header.stamp
            self._path.header.frame_id = frame_id
            self._path.poses.append(pose)
            if self.max_path_len > 0 and len(self._path.poses) > self.max_path_len:
                # drop oldest
                self._path.poses = self._path.poses[-self.max_path_len :]
            path_msg = self._path

        self._path_pub.publish(path_msg)


def main():
    rospy.init_node("gnss_degradation_viz")
    _ = GnssDegradationViz()
    rospy.spin()


if __name__ == "__main__":
    main()

