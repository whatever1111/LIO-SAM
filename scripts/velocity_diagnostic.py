#!/usr/bin/env python3
"""
速度诊断脚本 - 分析 Large velocity 问题的根本原因
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from lio_sam.msg import MappingStatus
import numpy as np
from collections import deque

class VelocityDiagnostic:
    def __init__(self):
        # Buffers
        self.odom_buffer = deque(maxlen=100)
        self.imu_buffer = deque(maxlen=1000)
        self.status_buffer = deque(maxlen=500)

        # Statistics
        self.odom_count = 0
        self.large_vel_events = []
        self.position_jumps = []
        self.imu_dt_issues = []

        # Subscribers
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental_status',
                        MappingStatus, self.status_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        rospy.loginfo("="*70)
        rospy.loginfo("Velocity Diagnostic Started")
        rospy.loginfo("Monitoring: /lio_sam/mapping/odometry_incremental")
        rospy.loginfo("Threshold: 30 m/s (108 km/h)")
        rospy.loginfo("="*70)

        self.last_imu_time = None

    def imu_callback(self, msg):
        """监控 IMU 时间戳间隔"""
        t = msg.header.stamp.to_sec()

        if self.last_imu_time is not None:
            dt = t - self.last_imu_time
            # 正常 IMU 频率 200Hz -> dt = 0.005s
            if dt > 0.02:  # 超过 20ms 间隔
                self.imu_dt_issues.append({
                    'time': t,
                    'dt': dt,
                    'expected': 0.005
                })
                rospy.logwarn("IMU gap: dt=%.3fs (expected ~0.005s) at t=%.3f",
                             dt, t)

        self.last_imu_time = t
        self.imu_buffer.append({
            'time': t,
            'acc': np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z]),
            'gyro': np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
        })

    def odom_callback(self, msg):
        """分析里程计速度"""
        self.odom_count += 1
        t = msg.header.stamp.to_sec()

        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        is_degenerate = False
        if len(self.status_buffer) > 0:
            best_dt = float('inf')
            for s in self.status_buffer:
                dt = abs(s['time'] - t)
                if dt < best_dt:
                    best_dt = dt
                    is_degenerate = s['degenerate']
            if best_dt > 0.05:
                is_degenerate = False

        self.odom_buffer.append({
            'time': t,
            'position': pos,
            'degenerate': is_degenerate
        })

        if len(self.odom_buffer) >= 2:
            self.analyze()

    def status_callback(self, msg):
        self.status_buffer.append({
            'time': msg.header.stamp.to_sec(),
            'degenerate': bool(msg.is_degenerate)
        })

    def analyze(self):
        """分析速度和位置跳变"""
        prev = self.odom_buffer[-2]
        curr = self.odom_buffer[-1]

        dt = curr['time'] - prev['time']
        if dt <= 0:
            return

        dp = curr['position'] - prev['position']
        distance = np.linalg.norm(dp)
        velocity = distance / dt

        # 检测位置跳变 (会导致大速度估计)
        if distance > 1.0:  # 1米跳变
            self.position_jumps.append({
                'time': curr['time'],
                'distance': distance,
                'dt': dt,
                'velocity': velocity,
                'degenerate': curr['degenerate']
            })

            rospy.logwarn("="*60)
            rospy.logwarn("POSITION JUMP DETECTED!")
            rospy.logwarn("  Distance: %.3f m in %.3f s", distance, dt)
            rospy.logwarn("  Implied velocity: %.2f m/s (%.1f km/h)",
                         velocity, velocity*3.6)
            rospy.logwarn("  Degenerate: %s", curr['degenerate'])
            rospy.logwarn("  From: [%.2f, %.2f, %.2f]",
                         prev['position'][0], prev['position'][1], prev['position'][2])
            rospy.logwarn("  To:   [%.2f, %.2f, %.2f]",
                         curr['position'][0], curr['position'][1], curr['position'][2])

            # 检查是否有 IMU 间隔问题
            recent_imu_issues = [i for i in self.imu_dt_issues
                                if abs(i['time'] - curr['time']) < 0.5]
            if recent_imu_issues:
                rospy.logwarn("  IMU gaps near this jump: %d", len(recent_imu_issues))

            rospy.logwarn("="*60)

        # 检测超过30 m/s阈值的速度
        if velocity > 30:
            self.large_vel_events.append({
                'time': curr['time'],
                'velocity': velocity,
                'distance': distance,
                'dt': dt
            })

            rospy.logerr("="*60)
            rospy.logerr("LARGE VELOCITY EVENT #%d", len(self.large_vel_events))
            rospy.logerr("  Velocity: %.2f m/s (%.1f km/h) > 30 m/s threshold",
                        velocity, velocity*3.6)
            rospy.logerr("  This will trigger IMU-preintegration reset!")
            rospy.logerr("="*60)

        # 定期统计
        if self.odom_count % 100 == 0:
            self.print_summary()

    def print_summary(self):
        """打印统计摘要"""
        rospy.loginfo("\n--- Velocity Diagnostic Summary ---")
        rospy.loginfo("Total odometry messages: %d", self.odom_count)
        rospy.loginfo("Position jumps (>1m): %d", len(self.position_jumps))
        rospy.loginfo("Large velocity events (>30m/s): %d", len(self.large_vel_events))
        rospy.loginfo("IMU timing issues: %d", len(self.imu_dt_issues))

        if self.position_jumps:
            max_jump = max(self.position_jumps, key=lambda x: x['distance'])
            rospy.loginfo("Largest jump: %.2f m (%.1f m/s)",
                         max_jump['distance'], max_jump['velocity'])

        if self.large_vel_events:
            max_vel = max(self.large_vel_events, key=lambda x: x['velocity'])
            rospy.loginfo("Max velocity: %.2f m/s (%.1f km/h)",
                         max_vel['velocity'], max_vel['velocity']*3.6)

if __name__ == '__main__':
    rospy.init_node('velocity_diagnostic')

    diagnostic = VelocityDiagnostic()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("\n\n" + "="*70)
        rospy.loginfo("FINAL DIAGNOSTIC REPORT")
        rospy.loginfo("="*70)
        diagnostic.print_summary()

        if diagnostic.large_vel_events:
            rospy.loginfo("\nROOT CAUSE ANALYSIS:")

            # 分析原因
            jump_related = sum(1 for e in diagnostic.large_vel_events
                              if any(abs(j['time'] - e['time']) < 0.1
                                    for j in diagnostic.position_jumps))

            if jump_related > 0:
                rospy.loginfo("  - %d/%d events related to position jumps",
                             jump_related, len(diagnostic.large_vel_events))
                rospy.loginfo("  -> Check LiDAR odometry stability")
                rospy.loginfo("  -> ECEF->ENU conversion may have issues")

            if diagnostic.imu_dt_issues:
                rospy.loginfo("  - IMU timing gaps detected: %d",
                             len(diagnostic.imu_dt_issues))
                rospy.loginfo("  -> Check IMU data continuity")
        else:
            rospy.loginfo("\nNo large velocity events detected!")
