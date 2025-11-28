#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from lio_sam.msg import cloud_info
import numpy as np
from collections import deque

class PipelineMonitor:
    def __init__(self):
        # 存储时间戳以跟踪数据流
        self.timestamps = {}
        self.processing_times = deque(maxlen=100)

        # 订阅所有中间话题
        rospy.Subscriber('/lidar_points', PointCloud2, self.lidar_callback)
        rospy.Subscriber('/lio_sam/deskew/cloud_info', cloud_info, self.deskew_callback)
        rospy.Subscriber('/lio_sam/feature/cloud_info', cloud_info, self.feature_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.odom_callback)

        self.frame_count = 0
        rospy.loginfo("Pipeline Monitor Started - Tracking data flow through LIO-SAM")

    def lidar_callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        recv_time = rospy.Time.now().to_sec()

        # 记录新帧
        self.timestamps[stamp] = {
            'lidar_stamp': stamp,
            'lidar_recv': recv_time,
            'lidar_delay': (recv_time - stamp) * 1000
        }

    def deskew_callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        recv_time = rospy.Time.now().to_sec()

        if stamp in self.timestamps:
            self.timestamps[stamp]['deskew_recv'] = recv_time
            self.timestamps[stamp]['deskew_delay'] = (recv_time - stamp) * 1000
            self.timestamps[stamp]['imageProj_time'] = (recv_time - self.timestamps[stamp]['lidar_recv']) * 1000

    def feature_callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        recv_time = rospy.Time.now().to_sec()

        if stamp in self.timestamps:
            self.timestamps[stamp]['feature_recv'] = recv_time
            self.timestamps[stamp]['feature_delay'] = (recv_time - stamp) * 1000
            if 'deskew_recv' in self.timestamps[stamp]:
                self.timestamps[stamp]['feature_extract_time'] = (recv_time - self.timestamps[stamp]['deskew_recv']) * 1000

    def odom_callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        recv_time = rospy.Time.now().to_sec()

        if stamp in self.timestamps:
            data = self.timestamps[stamp]
            data['odom_recv'] = recv_time
            data['odom_delay'] = (recv_time - stamp) * 1000

            if 'feature_recv' in data:
                data['map_opt_time'] = (recv_time - data['feature_recv']) * 1000

            # 打印完整的流水线分析
            self.frame_count += 1
            if self.frame_count % 10 == 1:  # 每10帧打印一次
                self.print_pipeline_analysis(data)

            # 清理旧数据
            self.cleanup_old_timestamps(stamp)

    def print_pipeline_analysis(self, data):
        rospy.loginfo("="*60)
        rospy.loginfo("PIPELINE TIMING ANALYSIS (all times in ms):")
        rospy.loginfo("-"*60)

        # 端到端延迟
        total_delay = data.get('odom_delay', 0)
        rospy.loginfo(f"📊 END-TO-END DELAY: {total_delay:.1f}ms")
        rospy.loginfo("")

        # 各阶段处理时间
        rospy.loginfo("Processing Times:")
        if 'imageProj_time' in data:
            rospy.loginfo(f"  1. ImageProjection:    {data['imageProj_time']:>7.1f}ms")

        if 'feature_extract_time' in data:
            rospy.loginfo(f"  2. FeatureExtraction:  {data['feature_extract_time']:>7.1f}ms")

        if 'map_opt_time' in data:
            rospy.loginfo(f"  3. MapOptimization:    {data['map_opt_time']:>7.1f}ms")

        # 计算总处理时间
        proc_total = sum([
            data.get('imageProj_time', 0),
            data.get('feature_extract_time', 0),
            data.get('map_opt_time', 0)
        ])
        rospy.loginfo(f"  Total Processing:      {proc_total:>7.1f}ms")

        # 未解释的延迟
        unexplained = total_delay - proc_total
        if unexplained > 10:
            rospy.logwarn(f"  ⚠️ Unexplained delay:   {unexplained:>7.1f}ms")

        # 累积延迟
        rospy.loginfo("")
        rospy.loginfo("Cumulative Delays (from data timestamp):")
        rospy.loginfo(f"  After ImageProjection:  {data.get('deskew_delay', 0):>7.1f}ms")
        rospy.loginfo(f"  After FeatureExtraction:{data.get('feature_delay', 0):>7.1f}ms")
        rospy.loginfo(f"  After MapOptimization:  {data.get('odom_delay', 0):>7.1f}ms")

        rospy.loginfo("="*60)

    def cleanup_old_timestamps(self, current_stamp):
        # 删除5秒前的数据
        cutoff = current_stamp - 5.0
        old_keys = [k for k in self.timestamps.keys() if k < cutoff]
        for k in old_keys:
            del self.timestamps[k]

if __name__ == '__main__':
    rospy.init_node('pipeline_monitor')
    monitor = PipelineMonitor()
    rospy.spin()