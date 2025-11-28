#!/usr/bin/env python3
"""
深度诊断大速度问题的根本原因
运行时需要同时运行 LIO-SAM
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque

class DeepVelocityDiagnosis:
    def __init__(self):
        # Data buffers
        self.odom_incremental = deque(maxlen=200)
        self.odom_mapping = deque(maxlen=200)
        self.imu_data = deque(maxlen=2000)

        # Statistics
        self.jump_events = []
        self.frame_count = 0

        # Subscribers
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_incremental_cb)
        rospy.Subscriber('/lio_sam/mapping/odometry',
                        Odometry, self.odom_mapping_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)

        rospy.loginfo("="*60)
        rospy.loginfo("Deep Velocity Diagnosis Started")
        rospy.loginfo("="*60)

    def imu_cb(self, msg):
        self.imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc': np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z])
        })

    def odom_mapping_cb(self, msg):
        """来自因子图优化后的位姿"""
        pass  # 暂不处理

    def odom_incremental_cb(self, msg):
        """来自 scan-to-map 匹配的增量里程计"""
        self.frame_count += 1
        t = msg.header.stamp.to_sec()

        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # 检查协方差(退化标志)
        is_degenerate = (msg.pose.covariance[0] == 1.0)

        self.odom_incremental.append({
            'time': t,
            'pos': pos,
            'degenerate': is_degenerate
        })

        if len(self.odom_incremental) >= 2:
            self.analyze_jump()

    def analyze_jump(self):
        prev = self.odom_incremental[-2]
        curr = self.odom_incremental[-1]

        dt = curr['time'] - prev['time']
        if dt <= 0:
            return

        dp = curr['pos'] - prev['pos']
        distance = np.linalg.norm(dp)
        velocity = distance / dt

        # 检测大速度事件
        if velocity > 30:  # 30 m/s threshold
            self.jump_events.append({
                'frame': self.frame_count,
                'time': curr['time'],
                'velocity': velocity,
                'distance': distance,
                'dt': dt,
                'pos_from': prev['pos'].copy(),
                'pos_to': curr['pos'].copy(),
                'degenerate': curr['degenerate']
            })

            rospy.logerr("="*60)
            rospy.logerr("LARGE VELOCITY EVENT #%d", len(self.jump_events))
            rospy.logerr("  Frame: %d", self.frame_count)
            rospy.logerr("  Velocity: %.1f m/s (%.0f km/h)", velocity, velocity*3.6)
            rospy.logerr("  Distance: %.2f m in %.3f s", distance, dt)
            rospy.logerr("  Degenerate: %s", curr['degenerate'])

            # 分析跳变分量
            rospy.logerr("  Position change:")
            rospy.logerr("    dX: %.2f m", dp[0])
            rospy.logerr("    dY: %.2f m", dp[1])
            rospy.logerr("    dZ: %.2f m", dp[2])

            # 找到主要跳变方向
            main_axis = np.argmax(np.abs(dp))
            axis_names = ['X', 'Y', 'Z']
            rospy.logerr("  Main jump axis: %s (%.2f m)",
                        axis_names[main_axis], dp[main_axis])

            # 检查IMU数据
            self.check_imu_during_jump(prev['time'], curr['time'], dp, dt)

            rospy.logerr("="*60)

        # 定期报告
        if self.frame_count % 100 == 0:
            self.report_summary()

    def check_imu_during_jump(self, t_start, t_end, dp, dt):
        """检查跳变期间的IMU数据"""
        # 获取这段时间内的IMU
        imu_in_window = [d for d in self.imu_data
                       if t_start <= d['time'] <= t_end]

        if not imu_in_window:
            rospy.logerr("  IMU: No data in this window!")
            return

        # 计算平均加速度
        accs = np.array([d['acc'] for d in imu_in_window])
        mean_acc = np.mean(accs, axis=0)

        # 去除重力后的加速度
        gravity = np.array([0, 0, 9.81])
        linear_acc = mean_acc - gravity

        # 根据加速度估计的位移 (s = 0.5 * a * t^2)
        expected_dp = 0.5 * linear_acc * dt * dt

        rospy.logerr("  IMU analysis (%d samples):", len(imu_in_window))
        rospy.logerr("    Mean acc (no gravity): [%.2f, %.2f, %.2f] m/s²",
                    linear_acc[0], linear_acc[1], linear_acc[2])
        rospy.logerr("    Expected displacement: [%.4f, %.4f, %.4f] m",
                    expected_dp[0], expected_dp[1], expected_dp[2])
        rospy.logerr("    Actual displacement:   [%.2f, %.2f, %.2f] m",
                    dp[0], dp[1], dp[2])

        # 检查是否合理
        ratio = np.linalg.norm(dp) / (np.linalg.norm(expected_dp) + 1e-6)
        if ratio > 100:
            rospy.logerr("    ⚠️  Actual/Expected ratio: %.0f (IMU不支持这个位移!)", ratio)

    def report_summary(self):
        rospy.loginfo("\n--- Summary at frame %d ---", self.frame_count)
        rospy.loginfo("Large velocity events: %d", len(self.jump_events))

        if self.jump_events:
            # 统计退化相关性
            degenerate_events = sum(1 for e in self.jump_events if e['degenerate'])
            rospy.loginfo("  Degenerate-related: %d/%d (%.0f%%)",
                         degenerate_events, len(self.jump_events),
                         100*degenerate_events/len(self.jump_events))

            # 主要跳变方向统计
            main_axes = []
            for e in self.jump_events:
                dp = e['pos_to'] - e['pos_from']
                main_axes.append(np.argmax(np.abs(dp)))

            from collections import Counter
            axis_counts = Counter(main_axes)
            axis_names = ['X', 'Y', 'Z']
            rospy.loginfo("  Main jump directions: %s",
                         {axis_names[k]: v for k, v in axis_counts.items()})

if __name__ == '__main__':
    rospy.init_node('deep_velocity_diagnosis')
    diag = DeepVelocityDiagnosis()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("FINAL REPORT")
        rospy.loginfo("="*60)
        rospy.loginfo("Total frames: %d", diag.frame_count)
        rospy.loginfo("Total jump events: %d", len(diag.jump_events))

        if diag.jump_events:
            # 分析所有事件
            velocities = [e['velocity'] for e in diag.jump_events]
            rospy.loginfo("Velocity range: %.1f - %.1f m/s",
                         min(velocities), max(velocities))

            # 时间分布
            if len(diag.jump_events) > 1:
                times = [e['time'] for e in diag.jump_events]
                rospy.loginfo("Time range: %.1f - %.1f s",
                             times[0] - times[0], times[-1] - times[0])
