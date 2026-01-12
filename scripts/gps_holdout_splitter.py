#!/usr/bin/env python3
"""
GPS hold-out splitter (leave-one-out style evaluation helper).

Subscribes to an input nav_msgs/Odometry GPS topic (default: /odometry/gps) and publishes:
  - train_topic: used by LIO-SAM (fusion)
  - test_topic:  held-out reference for evaluation (NOT fed into fusion)

This enables "unbiased" evaluation in GNSS-good segments by comparing fusion output against
GPS samples that were not used to build the factor graph.
"""

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class GpsHoldoutSplitter:
    def __init__(self):
        self.input_topic = str(rospy.get_param("~input_topic", "/odometry/gps"))
        self.train_topic = str(rospy.get_param("~train_topic", "/odometry/gps_train"))
        self.test_topic = str(rospy.get_param("~test_topic", "/odometry/gps_test"))

        self.every_n = int(rospy.get_param("~every_n", 5))
        if self.every_n < 2:
            rospy.logwarn("~every_n=%d is too small; forcing to 2", self.every_n)
            self.every_n = 2

        # If true: only hold out when GNSS is GOOD; when degraded, forward all to train.
        self.holdout_only_when_gnss_good = bool(rospy.get_param("~holdout_only_when_gnss_good", True))
        self.degraded_topic = str(rospy.get_param("~degraded_topic", "/gnss_degraded"))

        self._is_degraded = False
        self._count = 0

        self._train_pub = rospy.Publisher(self.train_topic, Odometry, queue_size=200)
        self._test_pub = rospy.Publisher(self.test_topic, Odometry, queue_size=200)

        rospy.Subscriber(self.degraded_topic, Bool, self._degraded_cb, queue_size=200)
        rospy.Subscriber(self.input_topic, Odometry, self._gps_cb, queue_size=200)

        rospy.loginfo("GPS holdout splitter started")
        rospy.loginfo("  input_topic:  %s", self.input_topic)
        rospy.loginfo("  train_topic:  %s", self.train_topic)
        rospy.loginfo("  test_topic:   %s", self.test_topic)
        rospy.loginfo("  every_n:      %d (1/%d -> test, rest -> train)", self.every_n, self.every_n)
        rospy.loginfo("  holdout_only_when_gnss_good: %s", "true" if self.holdout_only_when_gnss_good else "false")

    def _degraded_cb(self, msg):
        self._is_degraded = bool(msg.data)

    def _gps_cb(self, msg):
        self._count += 1

        if self.holdout_only_when_gnss_good and self._is_degraded:
            self._train_pub.publish(msg)
            return

        # Deterministic split by sequence count:
        # - every Nth message -> test
        # - others -> train
        if (self._count % self.every_n) == 0:
            self._test_pub.publish(msg)
        else:
            self._train_pub.publish(msg)


def main():
    rospy.init_node("gps_holdout_splitter")
    _ = GpsHoldoutSplitter()
    rospy.spin()


if __name__ == "__main__":
    main()

