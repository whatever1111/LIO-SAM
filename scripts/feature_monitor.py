#!/usr/bin/env python3
"""
Real-time Feature Extraction and Scan Matching Monitor
Monitors the quality of feature extraction and scan matching for Livox LiDAR
"""

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time

class FeatureMonitor:
    def __init__(self):
        rospy.init_node('feature_monitor', anonymous=True)

        self.corner_count = 0
        self.surface_count = 0
        self.last_pos = None
        self.last_time = None
        self.velocity_history = []
        self.feature_history = []

        # Subscribers
        rospy.Subscriber('/lio_sam/feature/cloud_corner', PointCloud2, self.corner_callback)
        rospy.Subscriber('/lio_sam/feature/cloud_surface', PointCloud2, self.surface_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odom_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental', Odometry, self.odom_incr_callback)

        self.last_print_time = time.time()
        self.corner_received = False
        self.surface_received = False

        print("=" * 70)
        print("Feature Extraction and Scan Matching Monitor")
        print("=" * 70)
        print("Monitoring feature counts and velocity...")
        print()

    def corner_callback(self, msg):
        self.corner_count = msg.width * msg.height
        self.corner_received = True
        self.try_print_status()

    def surface_callback(self, msg):
        self.surface_count = msg.width * msg.height
        self.surface_received = True
        self.try_print_status()

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        current_time = msg.header.stamp.to_sec()

        if self.last_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0.001:
                dx = pos.x - self.last_pos.x
                dy = pos.y - self.last_pos.y
                dz = pos.z - self.last_pos.z
                velocity = np.sqrt(dx*dx + dy*dy + dz*dz) / dt

                self.velocity_history.append(velocity)
                if len(self.velocity_history) > 100:
                    self.velocity_history.pop(0)

        self.last_pos = pos
        self.last_time = current_time

    def odom_incr_callback(self, msg):
        # Check degenerate flag
        is_degenerate = msg.pose.covariance[0] == 1
        if is_degenerate:
            print("[WARN] Degenerate scan detected!")

    def try_print_status(self):
        current_time = time.time()
        if current_time - self.last_print_time < 1.0:
            return

        if not (self.corner_received and self.surface_received):
            return

        self.last_print_time = current_time
        self.corner_received = False
        self.surface_received = False

        # Feature status
        corner_ok = "OK" if self.corner_count >= 5 else "LOW"
        surface_ok = "OK" if self.surface_count >= 50 else "LOW"

        # Velocity stats
        if len(self.velocity_history) > 0:
            vel_mean = np.mean(self.velocity_history[-10:]) if len(self.velocity_history) >= 10 else np.mean(self.velocity_history)
            vel_max = np.max(self.velocity_history[-10:]) if len(self.velocity_history) >= 10 else np.max(self.velocity_history)
            vel_status = "SPIKE!" if vel_max > 10 else "OK"
        else:
            vel_mean = 0
            vel_max = 0
            vel_status = "N/A"

        # Position
        if self.last_pos:
            pos_str = f"[{self.last_pos.x:.2f}, {self.last_pos.y:.2f}, {self.last_pos.z:.2f}]"
        else:
            pos_str = "N/A"

        # Print status
        print(f"Corner: {self.corner_count:5d} [{corner_ok}] | Surface: {self.surface_count:5d} [{surface_ok}] | "
              f"Vel: {vel_mean:.2f} m/s (max: {vel_max:.2f}) [{vel_status}] | Pos: {pos_str}")

        # Store history
        self.feature_history.append({
            'time': current_time,
            'corner': self.corner_count,
            'surface': self.surface_count,
            'vel_mean': vel_mean,
            'vel_max': vel_max
        })

        # Warnings
        if self.corner_count < 5:
            print(f"  [WARN] Edge features too low! Need >= 5, got {self.corner_count}")
        if self.surface_count < 50:
            print(f"  [WARN] Surface features too low! Need >= 50, got {self.surface_count}")
        if vel_max > 10:
            print(f"  [ALERT] Velocity spike detected: {vel_max:.2f} m/s!")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        monitor = FeatureMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
