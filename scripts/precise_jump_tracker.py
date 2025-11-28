#!/usr/bin/env python3
"""
精确追踪速度跳变的根本原因
监控所有关键变量的变化
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from collections import deque
import time

class PreciseJumpTracker:
    def __init__(self):
        # 历史数据
        self.odom_history = deque(maxlen=50)
        self.imu_odom_history = deque(maxlen=50)
        self.feature_history = deque(maxlen=50)

        self.frame_count = 0
        self.jump_events = []

        # 上一帧数据
        self.last_odom = None
        self.last_imu_odom = None
        self.last_corner_count = 0
        self.last_surf_count = 0

        # 订阅多个关键topic
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_incr_cb)
        rospy.Subscriber('/lio_sam/mapping/odometry',
                        Odometry, self.odom_mapping_cb)
        rospy.Subscriber('/odometry/imu_incremental',
                        Odometry, self.imu_odom_cb)
        rospy.Subscriber('/lio_sam/feature/cloud_corner',
                        PointCloud2, self.corner_cb)
        rospy.Subscriber('/lio_sam/feature/cloud_surface',
                        PointCloud2, self.surf_cb)

        rospy.loginfo("="*70)
        rospy.loginfo("精确跳变追踪器 - 监控所有关键变量")
        rospy.loginfo("="*70)

    def corner_cb(self, msg):
        count = msg.width * msg.height
        change = count - self.last_corner_count if self.last_corner_count > 0 else 0
        self.last_corner_count = count
        self.feature_history.append({
            'time': msg.header.stamp.to_sec(),
            'type': 'corner',
            'count': count,
            'change': change
        })

    def surf_cb(self, msg):
        count = msg.width * msg.height
        change = count - self.last_surf_count if self.last_surf_count > 0 else 0
        self.last_surf_count = count
        self.feature_history.append({
            'time': msg.header.stamp.to_sec(),
            'type': 'surf',
            'count': count,
            'change': change
        })

        # 检测特征点数量剧变
        if abs(change) > 500:
            rospy.logwarn("特征点数量剧变: Surface %+d (现在: %d)", change, count)

    def imu_odom_cb(self, msg):
        """IMU预积分的里程计 - 这是scan-to-map的初始猜测来源"""
        t = msg.header.stamp.to_sec()
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        if self.last_imu_odom is not None:
            dt = t - self.last_imu_odom['time']
            if dt > 0:
                dp = pos - self.last_imu_odom['pos']
                vel = np.linalg.norm(dp) / dt

                if vel > 10:
                    rospy.logerr(">>> IMU预积分速度异常: %.1f m/s", vel)
                    rospy.logerr("    位移: [%.2f, %.2f, %.2f] in %.3fs",
                                dp[0], dp[1], dp[2], dt)

        self.last_imu_odom = {'time': t, 'pos': pos}
        self.imu_odom_history.append({'time': t, 'pos': pos.copy()})

    def odom_mapping_cb(self, msg):
        """因子图优化后的里程计"""
        pass

    def odom_incr_cb(self, msg):
        """增量里程计 - scan-to-map匹配的直接输出"""
        self.frame_count += 1
        t = msg.header.stamp.to_sec()

        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # 检查退化标志
        is_degenerate = (msg.pose.covariance[0] == 1.0)

        vel = 0
        if self.last_odom is not None:
            dt = t - self.last_odom['time']
            if dt > 0:
                dp = pos - self.last_odom['pos']
                dist = np.linalg.norm(dp)
                vel = dist / dt

                # 检测跳变
                if vel > 10:
                    self.analyze_jump_cause(t, vel, dp, dt, is_degenerate, pos)

        self.last_odom = {'time': t, 'pos': pos, 'degenerate': is_degenerate}
        self.odom_history.append({'time': t, 'pos': pos.copy(), 'vel': vel})

        # 定期报告
        if self.frame_count % 100 == 0:
            rospy.loginfo("--- Frame %d, 跳变事件: %d ---",
                         self.frame_count, len(self.jump_events))

    def analyze_jump_cause(self, t, vel, dp, dt, is_degenerate, current_pos):
        """详细分析跳变原因"""
        rospy.logerr("="*70)
        rospy.logerr("!!! 检测到跳变 #%d !!!", len(self.jump_events)+1)
        rospy.logerr("="*70)

        rospy.logerr("基本信息:")
        rospy.logerr("  时间: %.3f", t)
        rospy.logerr("  速度: %.1f m/s", vel)
        rospy.logerr("  位移: [%.2f, %.2f, %.2f] m in %.3fs", dp[0], dp[1], dp[2], dt)
        rospy.logerr("  退化标志: %s", is_degenerate)

        # 分析跳变方向
        main_axis = np.argmax(np.abs(dp))
        axis_names = ['X', 'Y', 'Z']
        rospy.logerr("  主要跳变轴: %s (%.2f m)", axis_names[main_axis], dp[main_axis])

        # 检查IMU预积分状态
        rospy.logerr("")
        rospy.logerr("IMU预积分状态:")
        if self.last_imu_odom:
            imu_pos = self.last_imu_odom['pos']
            rospy.logerr("  IMU预积分位置: [%.2f, %.2f, %.2f]",
                        imu_pos[0], imu_pos[1], imu_pos[2])
            rospy.logerr("  当前位置: [%.2f, %.2f, %.2f]",
                        current_pos[0], current_pos[1], current_pos[2])
            diff = current_pos - imu_pos
            rospy.logerr("  差异: [%.2f, %.2f, %.2f] (范数: %.2f m)",
                        diff[0], diff[1], diff[2], np.linalg.norm(diff))
        else:
            rospy.logerr("  无IMU预积分数据")

        # 检查特征点变化
        rospy.logerr("")
        rospy.logerr("特征点状态:")
        rospy.logerr("  当前Corner: %d", self.last_corner_count)
        rospy.logerr("  当前Surface: %d", self.last_surf_count)

        # 查找最近的特征点变化
        recent_changes = [f for f in self.feature_history
                         if t - f['time'] < 1.0 and abs(f.get('change', 0)) > 100]
        if recent_changes:
            rospy.logerr("  最近1秒内的大变化:")
            for f in recent_changes[-5:]:
                rospy.logerr("    %s: %+d (现在: %d)",
                            f['type'], f['change'], f['count'])

        # 分析可能的原因
        rospy.logerr("")
        rospy.logerr(">>> 可能原因分析:")

        if is_degenerate:
            rospy.logerr("  [1] 退化场景 - scan-to-map匹配信息不足")

        if self.last_surf_count > 3000 and self.last_corner_count < 200:
            rospy.logerr("  [2] 特征不均衡 - Surface多但Corner少，几何约束弱")

        if abs(dp[1]) > abs(dp[0]) and abs(dp[1]) > abs(dp[2]):
            rospy.logerr("  [3] Y方向主导跳变 - 可能是前进方向约束不足")

        if self.last_imu_odom:
            diff_norm = np.linalg.norm(current_pos - self.last_imu_odom['pos'])
            if diff_norm > 50:
                rospy.logerr("  [4] IMU预积分与LIO输出差异大(%.1fm) - 初始猜测可能有问题", diff_norm)

        rospy.logerr("="*70)

        self.jump_events.append({
            'time': t,
            'vel': vel,
            'dp': dp.copy(),
            'degenerate': is_degenerate,
            'corner': self.last_corner_count,
            'surf': self.last_surf_count
        })

def main():
    rospy.init_node('precise_jump_tracker')
    tracker = PreciseJumpTracker()

    rospy.loginfo("开始精确追踪... (Ctrl+C 停止)")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("")
        rospy.loginfo("="*70)
        rospy.loginfo("最终统计")
        rospy.loginfo("="*70)
        rospy.loginfo("总帧数: %d", tracker.frame_count)
        rospy.loginfo("跳变事件: %d", len(tracker.jump_events))

        if tracker.jump_events:
            # 统计分析
            vels = [e['vel'] for e in tracker.jump_events]
            degenerate_count = sum(1 for e in tracker.jump_events if e['degenerate'])

            rospy.loginfo("")
            rospy.loginfo("跳变统计:")
            rospy.loginfo("  速度范围: %.1f - %.1f m/s", min(vels), max(vels))
            rospy.loginfo("  退化相关: %d/%d (%.1f%%)",
                         degenerate_count, len(tracker.jump_events),
                         100*degenerate_count/len(tracker.jump_events))

            # 分析主要跳变方向
            x_jumps = sum(1 for e in tracker.jump_events
                         if np.argmax(np.abs(e['dp'])) == 0)
            y_jumps = sum(1 for e in tracker.jump_events
                         if np.argmax(np.abs(e['dp'])) == 1)
            z_jumps = sum(1 for e in tracker.jump_events
                         if np.argmax(np.abs(e['dp'])) == 2)
            rospy.loginfo("  跳变方向: X=%d, Y=%d, Z=%d", x_jumps, y_jumps, z_jumps)

if __name__ == '__main__':
    main()
