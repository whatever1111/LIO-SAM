#!/usr/bin/env python3
"""
诊断 IMU 预积分速度问题
监控实时速度并记录异常情况
"""

import rospy
from nav_msgs.msg import Odometry
import numpy as np
from collections import deque
import time

class VelocityDiagnostic:
    def __init__(self):
        self.odom_buffer = deque(maxlen=100)
        self.large_velocity_count = 0
        self.max_velocity = 0.0
        self.start_time = time.time()

        # Subscribe to LIO-SAM odometry
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_callback)

        rospy.loginfo("Velocity Diagnostic started")
        rospy.loginfo("Monitoring /lio_sam/mapping/odometry_incremental")
        rospy.loginfo("Threshold: 30 m/s (108 km/h)")

    def odom_callback(self, msg):
        # 计算速度（从线速度）
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        velocity = np.sqrt(vx**2 + vy**2 + vz**2)

        # 或者从位置变化计算速度
        timestamp = msg.header.stamp.to_sec()
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        self.odom_buffer.append({
            'time': timestamp,
            'position': position,
            'velocity': velocity
        })

        # 计算从位置变化得到的速度
        if len(self.odom_buffer) >= 2:
            prev = self.odom_buffer[-2]
            curr = self.odom_buffer[-1]
            dt = curr['time'] - prev['time']

            if dt > 0:
                dp = curr['position'] - prev['position']
                computed_vel = np.linalg.norm(dp) / dt

                # 检查是否超过阈值
                if computed_vel > 30.0:
                    self.large_velocity_count += 1
                    rospy.logwarn(
                        "Large velocity detected! "
                        f"Computed: {computed_vel:.2f} m/s ({computed_vel*3.6:.1f} km/h), "
                        f"Reported: {velocity:.2f} m/s, "
                        f"dt: {dt:.4f}s, "
                        f"dp: {np.linalg.norm(dp):.3f}m"
                    )

                    # 打印最近的位置历史
                    if len(self.odom_buffer) >= 5:
                        rospy.loginfo("Recent position history:")
                        for i, entry in enumerate(list(self.odom_buffer)[-5:]):
                            rospy.loginfo(f"  [{i}] t={entry['time']:.3f}, "
                                        f"pos=[{entry['position'][0]:.2f}, "
                                        f"{entry['position'][1]:.2f}, "
                                        f"{entry['position'][2]:.2f}]")

                # 更新最大速度
                if computed_vel > self.max_velocity:
                    self.max_velocity = computed_vel

        # 每10秒打印统计
        elapsed = time.time() - self.start_time
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            if hasattr(self, '_last_report_time'):
                if time.time() - self._last_report_time > 9:
                    self.print_stats()
                    self._last_report_time = time.time()
            else:
                self._last_report_time = time.time()

    def print_stats(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Velocity Statistics:")
        rospy.loginfo(f"  Max velocity: {self.max_velocity:.2f} m/s ({self.max_velocity*3.6:.1f} km/h)")
        rospy.loginfo(f"  Large velocity events: {self.large_velocity_count}")
        rospy.loginfo(f"  Buffer size: {len(self.odom_buffer)}")
        rospy.loginfo("=" * 60)

if __name__ == '__main__':
    rospy.init_node('velocity_diagnostic')

    diagnostic = VelocityDiagnostic()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("\nFinal Statistics:")
        diagnostic.print_stats()
