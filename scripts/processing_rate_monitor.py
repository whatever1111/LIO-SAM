#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from lio_sam.msg import cloud_info
from nav_msgs.msg import Odometry
import time
from collections import deque

class ProcessingRateMonitor:
    def __init__(self):
        # 记录每个节点的处理间隔
        self.lidar_times = deque(maxlen=100)
        self.deskew_times = deque(maxlen=100)
        self.feature_times = deque(maxlen=100)
        self.odom_times = deque(maxlen=100)

        # 记录消息时间戳
        self.lidar_stamps = deque(maxlen=100)
        self.deskew_stamps = deque(maxlen=100)
        self.feature_stamps = deque(maxlen=100)
        self.odom_stamps = deque(maxlen=100)

        rospy.Subscriber('/lidar_points', PointCloud2, self.lidar_cb)
        rospy.Subscriber('/lio_sam/deskew/cloud_info', cloud_info, self.deskew_cb)
        rospy.Subscriber('/lio_sam/feature/cloud_info', cloud_info, self.feature_cb)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odom_cb)

        self.count = 0
        rospy.Timer(rospy.Duration(5.0), self.print_stats)

    def lidar_cb(self, msg):
        now = rospy.Time.now().to_sec()
        self.lidar_times.append(now)
        self.lidar_stamps.append(msg.header.stamp.to_sec())

    def deskew_cb(self, msg):
        now = rospy.Time.now().to_sec()
        self.deskew_times.append(now)
        self.deskew_stamps.append(msg.header.stamp.to_sec())

    def feature_cb(self, msg):
        now = rospy.Time.now().to_sec()
        self.feature_times.append(now)
        self.feature_stamps.append(msg.header.stamp.to_sec())

    def odom_cb(self, msg):
        now = rospy.Time.now().to_sec()
        self.odom_times.append(now)
        self.odom_stamps.append(msg.header.stamp.to_sec())

    def calc_intervals(self, times):
        if len(times) < 2:
            return 0, 0, 0
        intervals = []
        for i in range(1, len(times)):
            intervals.append((times[i] - times[i-1]) * 1000)  # Convert to ms
        if intervals:
            return sum(intervals)/len(intervals), min(intervals), max(intervals)
        return 0, 0, 0

    def calc_unique_stamps(self, stamps):
        """Count unique timestamps (to detect if messages are dropped)"""
        if len(stamps) < 2:
            return 0
        unique = len(set(stamps))
        return unique

    def print_stats(self, event):
        rospy.loginfo("="*60)
        rospy.loginfo("PROCESSING RATE ANALYSIS")
        rospy.loginfo("-"*60)

        # 计算各节点的处理间隔
        lidar_avg, lidar_min, lidar_max = self.calc_intervals(self.lidar_times)
        deskew_avg, deskew_min, deskew_max = self.calc_intervals(self.deskew_times)
        feature_avg, feature_min, feature_max = self.calc_intervals(self.feature_times)
        odom_avg, odom_min, odom_max = self.calc_intervals(self.odom_times)

        # 计算实际频率
        lidar_hz = 1000/lidar_avg if lidar_avg > 0 else 0
        deskew_hz = 1000/deskew_avg if deskew_avg > 0 else 0
        feature_hz = 1000/feature_avg if feature_avg > 0 else 0
        odom_hz = 1000/odom_avg if odom_avg > 0 else 0

        rospy.loginfo("Topic                    | Hz    | Avg(ms) | Min(ms) | Max(ms) | Count")
        rospy.loginfo("-"*60)
        rospy.loginfo(f"/lidar_points            | {lidar_hz:5.1f} | {lidar_avg:7.1f} | {lidar_min:7.1f} | {lidar_max:7.1f} | {len(self.lidar_times)}")
        rospy.loginfo(f"/deskew/cloud_info       | {deskew_hz:5.1f} | {deskew_avg:7.1f} | {deskew_min:7.1f} | {deskew_max:7.1f} | {len(self.deskew_times)}")
        rospy.loginfo(f"/feature/cloud_info      | {feature_hz:5.1f} | {feature_avg:7.1f} | {feature_min:7.1f} | {feature_max:7.1f} | {len(self.feature_times)}")
        rospy.loginfo(f"/mapping/odometry        | {odom_hz:5.1f} | {odom_avg:7.1f} | {odom_min:7.1f} | {odom_max:7.1f} | {len(self.odom_times)}")

        # 计算丢失的帧
        lidar_unique = self.calc_unique_stamps(self.lidar_stamps)
        deskew_unique = self.calc_unique_stamps(self.deskew_stamps)
        feature_unique = self.calc_unique_stamps(self.feature_stamps)
        odom_unique = self.calc_unique_stamps(self.odom_stamps)

        rospy.loginfo("")
        rospy.loginfo("Unique timestamps (message dropping detection):")
        rospy.loginfo(f"  Lidar:   {lidar_unique} unique stamps in {len(self.lidar_stamps)} messages")
        rospy.loginfo(f"  Deskew:  {deskew_unique} unique stamps in {len(self.deskew_stamps)} messages")
        rospy.loginfo(f"  Feature: {feature_unique} unique stamps in {len(self.feature_stamps)} messages")
        rospy.loginfo(f"  Odom:    {odom_unique} unique stamps in {len(self.odom_stamps)} messages")

        # 关键发现
        if odom_hz > 0 and lidar_hz > 0:
            drop_rate = (1 - odom_hz/lidar_hz) * 100
            rospy.logwarn(f"⚠️ Message drop rate: {drop_rate:.1f}% (Odom only processes {odom_hz:.1f}/{lidar_hz:.1f} Hz)")

        if odom_avg > 150:
            rospy.logwarn(f"⚠️ MapOptimization interval: {odom_avg:.0f}ms (likely limited by mappingProcessInterval)")

        # 检查频率瓶颈
        if deskew_hz < lidar_hz * 0.9:
            rospy.logwarn(f"⚠️ ImageProjection bottleneck: {deskew_hz:.1f}Hz < {lidar_hz:.1f}Hz")
        if feature_hz < deskew_hz * 0.9:
            rospy.logwarn(f"⚠️ FeatureExtraction bottleneck: {feature_hz:.1f}Hz < {deskew_hz:.1f}Hz")
        if odom_hz < feature_hz * 0.9:
            rospy.logwarn(f"⚠️ MapOptimization bottleneck: {odom_hz:.1f}Hz < {feature_hz:.1f}Hz")

        rospy.loginfo("="*60)

if __name__ == '__main__':
    rospy.init_node('processing_rate_monitor')
    monitor = ProcessingRateMonitor()
    rospy.spin()