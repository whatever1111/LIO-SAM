#!/usr/bin/env python3
"""
监控GPS factor是否被添加
检查poseCovThreshold条件是否阻止GPS融合
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from lio_sam.msg import cloud_info
import threading

class GPSFactorMonitor:
    def __init__(self):
        rospy.init_node('gps_factor_monitor', anonymous=True)

        self.lock = threading.Lock()
        self.start_time = None

        # 统计
        self.gps_count = 0
        self.fusion_count = 0
        self.last_gps_pos = None
        self.gps_traveled = 0.0

        # 位置协方差 (从SLAM info获取)
        self.pose_cov_x = 0
        self.pose_cov_y = 0
        self.pose_cov_threshold = 25.0  # 从params.yaml

        rospy.Subscriber("/odometry/gps", Odometry, self.gps_callback)
        rospy.Subscriber("/lio_sam/mapping/odometry", Odometry, self.fusion_callback)
        rospy.Subscriber("/lio_sam/mapping/slam_info", cloud_info, self.slam_info_callback)

        print("=" * 80)
        print("GPS Factor 监控器")
        print("=" * 80)
        print("检查GPS factor是否被有效添加到图优化中")
        print()
        print("关键参数:")
        print(f"  poseCovThreshold: {self.pose_cov_threshold} m^2")
        print("  GPS每5米添加一次")
        print()
        print("如果poseCov始终 < 25，GPS factor永远不会添加！")
        print("-" * 80)

    def gps_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            # 计算GPS行驶距离
            if self.last_gps_pos is not None:
                dist = np.linalg.norm(pos[:2] - self.last_gps_pos[:2])
                self.gps_traveled += dist

            self.last_gps_pos = pos
            self.gps_count += 1

            # 获取GPS协方差
            gps_cov_x = msg.pose.covariance[0]
            gps_cov_y = msg.pose.covariance[7]

            rel_time = t - self.start_time

            if self.gps_count % 50 == 0:
                print(f"\n[GPS] t={rel_time:.1f}s, 总距离={self.gps_traveled:.1f}m, GPS协方差=[{gps_cov_x:.3f}, {gps_cov_y:.3f}]")

    def fusion_callback(self, msg):
        with self.lock:
            self.fusion_count += 1

    def slam_info_callback(self, msg):
        """获取SLAM状态信息，包括协方差"""
        # 注意：cloud_info消息可能不包含协方差
        # 这里主要是占位，实际协方差需要从其他地方获取
        pass

    def analyze(self):
        with self.lock:
            if self.start_time is None:
                return

            rel_time = rospy.Time.now().to_sec() - self.start_time

            print(f"\n{'='*80}")
            print(f"状态报告 t={rel_time:.1f}s")
            print(f"{'='*80}")
            print(f"  GPS消息数: {self.gps_count}")
            print(f"  融合消息数: {self.fusion_count}")
            print(f"  GPS总行驶距离: {self.gps_traveled:.1f}m")
            print(f"  预期GPS factor数: {int(self.gps_traveled / 5)} (每5米一个)")
            print()
            print("检查方法:")
            print("  1. 运行 'rostopic echo /lio_sam/mapping/odometry -n 1' 查看协方差")
            print("  2. 如果协方差一直很小，考虑降低 poseCovThreshold")
            print("  3. 或者检查LiDAR里程计是否过于自信")
            print(f"{'='*80}")

    def run(self):
        rate = rospy.Rate(0.2)  # 每5秒
        try:
            while not rospy.is_shutdown():
                self.analyze()
                rate.sleep()
        except KeyboardInterrupt:
            self.analyze()

if __name__ == '__main__':
    try:
        monitor = GPSFactorMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
