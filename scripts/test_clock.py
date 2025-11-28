#!/usr/bin/env python3
"""
Test script to diagnose clock synchronization issue with rosbag playback
"""

import rospy
from sensor_msgs.msg import PointCloud2
import time

class ClockTester:
    def __init__(self):
        rospy.init_node('clock_tester', anonymous=True)
        self.last_msg_time = None
        self.last_receive_time = None
        self.message_count = 0
        self.sub = rospy.Subscriber('/lidar_points', PointCloud2, self.callback)

    def callback(self, msg):
        # Get message timestamp
        msg_time = msg.header.stamp.to_sec()

        # Get current ROS time (should be simulation time when using --clock)
        current_ros_time = rospy.Time.now().to_sec()

        # Get system wall time
        wall_time = time.time()

        # Calculate time differences
        time_diff = current_ros_time - msg_time

        self.message_count += 1

        # Print diagnostics
        print(f"\n=== Message #{self.message_count} ===")
        print(f"Message timestamp:     {msg_time:.3f}")
        print(f"Current ROS time:      {current_ros_time:.3f}")
        print(f"Time difference:       {time_diff:.3f} seconds")
        print(f"Wall clock time:       {wall_time:.3f}")

        # Check for timeout condition
        if self.last_msg_time is not None:
            time_since_last = msg_time - self.last_msg_time
            ros_time_since_last = current_ros_time - self.last_receive_time

            print(f"Time since last msg:   {time_since_last:.3f} seconds (based on msg stamps)")
            print(f"ROS time since last:   {ros_time_since_last:.3f} seconds (based on ROS clock)")

            if ros_time_since_last > 1.0:
                print("[WARNING] Would trigger timeout in diagnostic script!")

        self.last_msg_time = msg_time
        self.last_receive_time = current_ros_time

if __name__ == '__main__':
    try:
        tester = ClockTester()
        print("Clock tester started. Monitoring /lidar_points topic...")
        print("Press Ctrl+C to stop.")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass