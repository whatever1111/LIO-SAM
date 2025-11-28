#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from lio_sam.msg import cloud_info
import time

class LatencyAnalyzer:
    def __init__(self):
        self.lidar_time = None
        self.cloud_info_time = None
        self.feature_time = None
        self.odom_time = None

        # 订阅各个阶段的话题
        rospy.Subscriber('/lidar_points', PointCloud2, self.lidar_callback)
        rospy.Subscriber('/lio_sam/deskew/cloud_info', cloud_info, self.deskew_callback)
        rospy.Subscriber('/lio_sam/feature/cloud_info', cloud_info, self.feature_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odom_callback)

        self.count = 0
        rospy.loginfo("Latency Analyzer started")

    def lidar_callback(self, msg):
        self.lidar_time = msg.header.stamp.to_sec()
        self.lidar_receive = rospy.Time.now().to_sec()

    def deskew_callback(self, msg):
        self.cloud_info_time = msg.header.stamp.to_sec()
        self.deskew_receive = rospy.Time.now().to_sec()

        if self.lidar_time:
            delay = (self.deskew_receive - self.cloud_info_time) * 1000
            rospy.loginfo_throttle(5, f"Deskew delay: {delay:.1f}ms (header time diff: {(self.cloud_info_time - self.lidar_time)*1000:.1f}ms)")

    def feature_callback(self, msg):
        self.feature_time = msg.header.stamp.to_sec()
        self.feature_receive = rospy.Time.now().to_sec()

    def odom_callback(self, msg):
        receive_time = rospy.Time.now().to_sec()
        msg_time = msg.header.stamp.to_sec()

        # 计算端到端延迟
        end_to_end_delay = (receive_time - msg_time) * 1000

        self.count += 1
        if self.count % 10 == 1:  # 每10帧打印一次
            rospy.loginfo("="*50)
            rospy.loginfo("Latency Analysis:")
            rospy.loginfo(f"  Message timestamp:     {msg_time:.3f}")
            rospy.loginfo(f"  Receive time:          {receive_time:.3f}")
            rospy.loginfo(f"  End-to-end delay:      {end_to_end_delay:.1f}ms")

            if self.lidar_time:
                rospy.loginfo(f"  Lidar->Odom time diff: {(msg_time - self.lidar_time)*1000:.1f}ms")
                rospy.loginfo(f"  Total processing:      {(receive_time - self.lidar_receive)*1000:.1f}ms")

            # 检查时间戳是否被保留
            if self.cloud_info_time:
                rospy.loginfo(f"  CloudInfo timestamp:   {self.cloud_info_time:.3f}")
                rospy.loginfo(f"  Timestamp preserved:   {abs(msg_time - self.cloud_info_time) < 0.001}")

            rospy.loginfo("="*50)

            # 这就是 realtime_plotter.py 计算的延迟
            rospy.logwarn(f"realtime_plotter.py would show: {end_to_end_delay:.1f}ms")

if __name__ == '__main__':
    rospy.init_node('latency_analyzer')
    analyzer = LatencyAnalyzer()
    rospy.spin()