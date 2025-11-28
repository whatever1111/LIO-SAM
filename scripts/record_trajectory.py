#!/usr/bin/env python3
"""
轨迹录制脚本
录制三条轨迹:
1. LIO-SAM融合轨迹 (/lio_sam/mapping/odometry)
2. GPS原始轨迹 (/odometry/gps)
3. 纯LIO轨迹 (待实现)
"""

import rospy
import rosbag
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import sys
import signal

class TrajectoryRecorder:
    def __init__(self, output_bag):
        self.bag = rosbag.Bag(output_bag, 'w')
        self.fusion_count = 0
        self.gps_count = 0
        self.degraded_count = 0
        self.bag_ready = False

        rospy.loginfo("Trajectory Recorder started, output: %s", output_bag)

        # Subscribe to topics after a short delay to ensure bag is ready
        rospy.Timer(rospy.Duration(0.5), self.start_recording, oneshot=True)

    def start_recording(self, event):
        self.bag_ready = True
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.fusion_callback)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_callback)
        rospy.Subscriber('/gnss_degraded', Bool, self.degraded_callback)
        rospy.loginfo("Recording started")

    def fusion_callback(self, msg):
        if not self.bag_ready:
            return
        try:
            self.bag.write('/lio_sam/mapping/odometry', msg)
            self.fusion_count += 1
            if self.fusion_count % 100 == 0:
                rospy.loginfo("Recorded %d fusion poses", self.fusion_count)
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Failed to write fusion: %s", str(e))

    def gps_callback(self, msg):
        if not self.bag_ready:
            return
        try:
            self.bag.write('/odometry/gps', msg)
            self.gps_count += 1
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Failed to write gps: %s", str(e))

    def degraded_callback(self, msg):
        if not self.bag_ready:
            return
        try:
            self.bag.write('/gnss_degraded', msg)
            self.degraded_count += 1
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Failed to write degraded: %s", str(e))

    def close(self):
        rospy.loginfo("Closing bag file...")
        rospy.loginfo("Total recorded - Fusion: %d, GPS: %d, Degraded status: %d",
                     self.fusion_count, self.gps_count, self.degraded_count)
        self.bag.close()

if __name__ == '__main__':
    rospy.init_node('trajectory_recorder')

    if len(sys.argv) < 2:
        output_bag = '/tmp/trajectory_evaluation.bag'
    else:
        output_bag = sys.argv[1]

    recorder = TrajectoryRecorder(output_bag)

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        rospy.loginfo("Received signal %d, shutting down gracefully...", sig)
        recorder.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        recorder.close()
