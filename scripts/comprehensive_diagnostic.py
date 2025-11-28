#!/usr/bin/env python3
"""
LIO-SAM 综合诊断工具
同时监控：
1. 速度异常
2. 时间戳同步问题
3. IMU/点云数据流
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
import numpy as np
from collections import deque
import time

class ComprehensiveDiagnostic:
    def __init__(self):
        # 数据缓存
        self.imu_buffer = deque(maxlen=1000)
        self.cloud_buffer = deque(maxlen=100)
        self.odom_buffer = deque(maxlen=100)

        # 统计
        self.large_velocity_count = 0
        self.max_velocity = 0.0
        self.timestamp_mismatch_count = 0
        self.start_time = time.time()

        # 消息计数
        self.imu_count = 0
        self.cloud_count = 0
        self.odom_count = 0

        # Subscribers
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/lidar_points', PointCloud2, self.cloud_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_callback)

        rospy.loginfo("="*70)
        rospy.loginfo("Comprehensive Diagnostic Started")
        rospy.loginfo("="*70)
        rospy.loginfo("Monitoring topics:")
        rospy.loginfo("  - /imu/data")
        rospy.loginfo("  - /lidar_points")
        rospy.loginfo("  - /lio_sam/mapping/odometry_incremental")
        rospy.loginfo("Velocity threshold: 30 m/s (108 km/h)")
        rospy.loginfo("="*70)

    def imu_callback(self, msg):
        self.imu_count += 1
        timestamp = msg.header.stamp.to_sec()
        self.imu_buffer.append({
            'time': timestamp,
            'seq': self.imu_count
        })

    def cloud_callback(self, msg):
        self.cloud_count += 1
        timestamp = msg.header.stamp.to_sec()

        self.cloud_buffer.append({
            'time': timestamp,
            'seq': self.cloud_count,
            'width': msg.width,
            'height': msg.height
        })

        # 检查时间戳同步
        self.check_timestamp_sync(timestamp)

        if self.cloud_count % 10 == 0:
            rospy.loginfo(f"Received {self.cloud_count} clouds, "
                         f"{self.imu_count} IMU msgs, "
                         f"{self.odom_count} odom msgs")

    def odom_callback(self, msg):
        self.odom_count += 1
        timestamp = msg.header.stamp.to_sec()

        # 提取位置和速度
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # 从twist获取速度
        twist_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        twist_speed = np.linalg.norm(twist_vel)

        self.odom_buffer.append({
            'time': timestamp,
            'position': position,
            'twist_velocity': twist_vel,
            'twist_speed': twist_speed
        })

        # 计算从位置变化得到的速度
        if len(self.odom_buffer) >= 2:
            self.check_velocity()

    def check_timestamp_sync(self, cloud_time):
        """检查点云和IMU时间戳同步"""
        if len(self.imu_buffer) == 0:
            rospy.logwarn(f"Cloud #{self.cloud_count}: No IMU data available!")
            return

        imu_times = [entry['time'] for entry in self.imu_buffer]
        imu_start = min(imu_times)
        imu_end = max(imu_times)

        # 假设点云扫描时间约0.1秒
        cloud_end = cloud_time + 0.1

        # 检查覆盖情况
        has_coverage = (imu_start <= cloud_time and imu_end >= cloud_end)

        if not has_coverage:
            self.timestamp_mismatch_count += 1
            rospy.logwarn(
                f"⚠️  Timestamp Mismatch #{self.timestamp_mismatch_count}!"
            )
            rospy.logwarn(
                f"    Cloud time: [{cloud_time:.6f}, {cloud_end:.6f}]"
            )
            rospy.logwarn(
                f"    IMU range:  [{imu_start:.6f}, {imu_end:.6f}]"
            )

            if imu_start > cloud_time:
                rospy.logwarn(
                    f"    ❌ IMU starts {imu_start - cloud_time:.3f}s AFTER cloud start"
                )
            if imu_end < cloud_end:
                rospy.logwarn(
                    f"    ❌ IMU ends {cloud_end - imu_end:.3f}s BEFORE cloud end"
                )

            rospy.logwarn(
                f"    IMU buffer: {len(self.imu_buffer)} msgs, "
                f"range: {imu_end - imu_start:.3f}s"
            )

    def check_velocity(self):
        """检查速度是否异常"""
        prev = self.odom_buffer[-2]
        curr = self.odom_buffer[-1]

        dt = curr['time'] - prev['time']

        if dt <= 0:
            rospy.logwarn(f"Non-positive dt: {dt:.6f}s")
            return

        # 从位置计算速度
        dp = curr['position'] - prev['position']
        computed_speed = np.linalg.norm(dp) / dt

        # 从twist获取的速度
        twist_speed = curr['twist_speed']

        # 检查计算的速度
        if computed_speed > 30.0:
            self.large_velocity_count += 1
            rospy.logerr("="*70)
            rospy.logerr(f"🚨 Large Velocity Detected! #{self.large_velocity_count}")
            rospy.logerr("="*70)
            rospy.logerr(f"  Computed from position: {computed_speed:.2f} m/s ({computed_speed*3.6:.1f} km/h)")
            rospy.logerr(f"  From twist:            {twist_speed:.2f} m/s ({twist_speed*3.6:.1f} km/h)")
            rospy.logerr(f"  Time delta:            {dt:.6f} s")
            rospy.logerr(f"  Position delta:        {np.linalg.norm(dp):.3f} m")
            rospy.logerr(f"  Delta vector:          [{dp[0]:.3f}, {dp[1]:.3f}, {dp[2]:.3f}]")

            # 显示最近的轨迹
            if len(self.odom_buffer) >= 5:
                rospy.logerr("\n  Recent trajectory:")
                for i, entry in enumerate(list(self.odom_buffer)[-5:]):
                    pos = entry['position']
                    rospy.logerr(f"    [{i}] t={entry['time']:.6f}, "
                               f"pos=[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}], "
                               f"twist_v={entry['twist_speed']:6.2f} m/s")
            rospy.logerr("="*70)

        # 更新最大速度
        if computed_speed > self.max_velocity:
            self.max_velocity = computed_speed

        # 检查twist速度
        if twist_speed > 30.0:
            rospy.logwarn(f"Twist velocity also large: {twist_speed:.2f} m/s")

    def print_stats(self):
        """打印统计信息"""
        elapsed = time.time() - self.start_time

        rospy.loginfo("")
        rospy.loginfo("="*70)
        rospy.loginfo("DIAGNOSTIC STATISTICS")
        rospy.loginfo("="*70)
        rospy.loginfo(f"Runtime: {elapsed:.1f} seconds")
        rospy.loginfo("")
        rospy.loginfo("Message counts:")
        rospy.loginfo(f"  IMU:         {self.imu_count} msgs ({self.imu_count/elapsed:.1f} Hz)")
        rospy.loginfo(f"  Point Cloud: {self.cloud_count} msgs ({self.cloud_count/elapsed:.1f} Hz)")
        rospy.loginfo(f"  Odometry:    {self.odom_count} msgs ({self.odom_count/elapsed:.1f} Hz)")
        rospy.loginfo("")
        rospy.loginfo("Velocity statistics:")
        rospy.loginfo(f"  Max velocity:        {self.max_velocity:.2f} m/s ({self.max_velocity*3.6:.1f} km/h)")
        rospy.loginfo(f"  Large velocity events: {self.large_velocity_count}")
        rospy.loginfo("")
        rospy.loginfo("Timestamp statistics:")
        rospy.loginfo(f"  Timestamp mismatches: {self.timestamp_mismatch_count}")

        if len(self.imu_buffer) > 0:
            imu_times = [e['time'] for e in self.imu_buffer]
            rospy.loginfo(f"  IMU time range: [{min(imu_times):.3f}, {max(imu_times):.3f}]")

        if len(self.cloud_buffer) > 0:
            cloud_times = [e['time'] for e in self.cloud_buffer]
            rospy.loginfo(f"  Cloud time range: [{min(cloud_times):.3f}, {max(cloud_times):.3f}]")

        rospy.loginfo("="*70)
        rospy.loginfo("")

if __name__ == '__main__':
    rospy.init_node('comprehensive_diagnostic')

    diagnostic = ComprehensiveDiagnostic()

    # 定期打印统计
    def timer_callback(event):
        diagnostic.print_stats()

    rospy.Timer(rospy.Duration(10), timer_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("\nShutting down...")
        rospy.loginfo("\n" + "="*70)
        rospy.loginfo("FINAL STATISTICS")
        rospy.loginfo("="*70)
        diagnostic.print_stats()
