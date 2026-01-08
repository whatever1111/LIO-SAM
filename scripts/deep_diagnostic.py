#!/usr/bin/env python3
"""
深度诊断 - 分析位置跳变和优化失败的原因
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from lio_sam.msg import cloud_info, MappingStatus
import numpy as np
from collections import deque

class DeepDiagnostic:
    def __init__(self):
        # 缓存
        self.odom_buffer = deque(maxlen=200)
        self.status_buffer = deque(maxlen=500)

        # 统计
        self.position_stuck_count = 0
        self.position_jump_count = 0
        self.optimization_success_count = 0
        self.optimization_fail_count = 0

        # Subscribers
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental_status',
                        MappingStatus, self.status_callback)

        # 订阅 cloud_info 来检查优化状态
        rospy.Subscriber('/lio_sam/feature/cloud_info',
                        cloud_info, self.cloud_info_callback)

        rospy.loginfo("="*70)
        rospy.loginfo("Deep Diagnostic Started")
        rospy.loginfo("="*70)

        self.msg_count = 0

    def cloud_info_callback(self, msg):
        """检查特征提取质量"""
        # cloud_info 包含特征点数量等信息
        pass

    def odom_callback(self, msg):
        self.msg_count += 1
        timestamp = msg.header.stamp.to_sec()

        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # 检查协方差 - degenerate标志
        is_degenerate = False
        if len(self.status_buffer) > 0:
            best_dt = float('inf')
            for s in self.status_buffer:
                dt = abs(s['time'] - timestamp)
                if dt < best_dt:
                    best_dt = dt
                    is_degenerate = s['degenerate']
            if best_dt > 0.05:
                is_degenerate = False

        self.odom_buffer.append({
            'time': timestamp,
            'position': position,
            'is_degenerate': is_degenerate,
            'seq': self.msg_count
        })

        if len(self.odom_buffer) >= 2:
            self.analyze_motion()

    def status_callback(self, msg):
        self.status_buffer.append({
            'time': msg.header.stamp.to_sec(),
            'degenerate': bool(msg.is_degenerate)
        })

    def analyze_motion(self):
        """分析运动特征"""
        prev = self.odom_buffer[-2]
        curr = self.odom_buffer[-1]

        dt = curr['time'] - prev['time']
        if dt <= 0:
            return

        dp = curr['position'] - prev['position']
        distance = np.linalg.norm(dp)
        speed = distance / dt

        # 检测位置停滞（几乎不动）
        if distance < 0.01:  # 1cm
            self.position_stuck_count += 1
            if self.position_stuck_count % 5 == 1:
                rospy.logwarn(f"⚠️  Position STUCK #{self.position_stuck_count}: "
                            f"Moved only {distance*100:.2f}cm in {dt:.3f}s, "
                            f"degenerate={curr['is_degenerate']}")

        # 检测位置跳变（速度过大）
        if speed > 30.0:
            self.position_jump_count += 1

            # 检查是否是从停滞恢复
            was_stuck = distance > 5.0  # 跳变超过5米

            rospy.logerr("="*70)
            rospy.logerr(f"🚨 Position JUMP #{self.position_jump_count}")
            rospy.logerr("="*70)
            rospy.logerr(f"  Speed: {speed:.2f} m/s ({speed*3.6:.1f} km/h)")
            rospy.logerr(f"  Distance: {distance:.3f} m in {dt:.3f}s")
            rospy.logerr(f"  Current degenerate: {curr['is_degenerate']}")
            rospy.logerr(f"  Previous degenerate: {prev['is_degenerate']}")

            # 检查最近是否有停滞
            if len(self.odom_buffer) >= 5:
                recent = list(self.odom_buffer)[-5:]
                stuck_frames = 0
                for i in range(len(recent)-1):
                    d = np.linalg.norm(recent[i+1]['position'] - recent[i]['position'])
                    if d < 0.01:
                        stuck_frames += 1

                if stuck_frames > 0:
                    rospy.logerr(f"  ⚠️  {stuck_frames}/4 recent frames were STUCK!")
                    rospy.logerr(f"  This jump likely from stuck → sudden update")

            rospy.logerr("="*70)

        # 定期统计
        if self.msg_count % 50 == 0:
            rospy.loginfo(f"\n--- Statistics (every 50 messages) ---")
            rospy.loginfo(f"Total odometry messages: {self.msg_count}")
            rospy.loginfo(f"Position stuck events: {self.position_stuck_count}")
            rospy.loginfo(f"Position jump events: {self.position_jump_count}")
            rospy.loginfo(f"Stuck ratio: {self.position_stuck_count/self.msg_count*100:.1f}%")

if __name__ == '__main__':
    rospy.init_node('deep_diagnostic')

    diagnostic = DeepDiagnostic()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("\n\nFinal Statistics:")
        rospy.loginfo(f"Total messages: {diagnostic.msg_count}")
        rospy.loginfo(f"Stuck events: {diagnostic.position_stuck_count} ({diagnostic.position_stuck_count/max(1,diagnostic.msg_count)*100:.1f}%)")
        rospy.loginfo(f"Jump events: {diagnostic.position_jump_count}")
