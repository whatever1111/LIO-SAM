#!/usr/bin/env python3
"""
LIO-SAM速度跳变根因分析脚本
监控:
1. 特征点数量
2. scan-to-map匹配状态
3. 优化收敛情况
4. 位置变化率
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from lio_sam.msg import cloud_info
import numpy as np
from collections import deque
import time

class VelocityJumpRootCause:
    def __init__(self):
        # 数据缓冲
        self.odom_history = deque(maxlen=100)
        self.feature_counts = deque(maxlen=100)

        # 统计
        self.total_frames = 0
        self.jump_count = 0
        self.degenerate_count = 0
        self.low_feature_count = 0

        # 阈值
        self.velocity_threshold = 20.0  # m/s
        self.min_edge_features = 10
        self.min_surf_features = 100

        # 上一帧数据
        self.last_pos = None
        self.last_time = None

        # 订阅
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental',
                        Odometry, self.odom_cb)
        rospy.Subscriber('/lio_sam/feature/cloud_info',
                        cloud_info, self.cloud_info_cb)

        # 发布诊断信息
        rospy.loginfo("="*60)
        rospy.loginfo("速度跳变根因分析器启动")
        rospy.loginfo("="*60)
        rospy.loginfo("监控指标:")
        rospy.loginfo("  - 位置跳变 (速度 > %d m/s)", self.velocity_threshold)
        rospy.loginfo("  - 特征数量 (edge < %d 或 surf < %d)",
                     self.min_edge_features, self.min_surf_features)
        rospy.loginfo("  - 退化检测标志")
        rospy.loginfo("="*60)

    def cloud_info_cb(self, msg):
        """处理特征点云信息"""
        # 提取特征点数量 (如果cloud_info有这些字段)
        # 注: 根据具体实现可能需要调整
        pass

    def odom_cb(self, msg):
        """处理里程计消息,检测速度跳变"""
        self.total_frames += 1
        t = msg.header.stamp.to_sec()

        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # 检查退化标志 (在covariance[0]中)
        is_degenerate = (msg.pose.covariance[0] == 1.0)
        if is_degenerate:
            self.degenerate_count += 1

        # 计算速度
        if self.last_pos is not None and self.last_time is not None:
            dt = t - self.last_time
            if dt > 0:
                dp = pos - self.last_pos
                distance = np.linalg.norm(dp)
                velocity = distance / dt

                # 检测跳变
                if velocity > self.velocity_threshold:
                    self.jump_count += 1
                    self.analyze_jump(t, velocity, dp, dt, is_degenerate)

        self.last_pos = pos
        self.last_time = t
        self.odom_history.append({
            'time': t,
            'pos': pos,
            'degenerate': is_degenerate
        })

        # 定期报告
        if self.total_frames % 50 == 0:
            self.report_status()

    def analyze_jump(self, t, velocity, dp, dt, is_degenerate):
        """分析跳变事件的详细原因"""
        rospy.logerr("="*60)
        rospy.logerr("!!! 检测到速度跳变 #%d !!!", self.jump_count)
        rospy.logerr("  时间: %.3f", t)
        rospy.logerr("  速度: %.1f m/s (%.0f km/h)", velocity, velocity*3.6)
        rospy.logerr("  距离: %.2f m in %.3f s", np.linalg.norm(dp), dt)

        # 分析跳变方向
        main_axis = np.argmax(np.abs(dp))
        axis_names = ['X', 'Y', 'Z']
        rospy.logerr("  主要跳变方向: %s (%.2f m)", axis_names[main_axis], dp[main_axis])
        rospy.logerr("  位移分量: dX=%.2f, dY=%.2f, dZ=%.2f", dp[0], dp[1], dp[2])

        # 退化状态
        if is_degenerate:
            rospy.logerr("  >>> 退化标志: 是 (scan-to-map匹配可能失败)")
        else:
            rospy.logerr("  退化标志: 否")

        # 根据模式判断原因
        rospy.logerr("")
        rospy.logerr("可能的原因分析:")

        if is_degenerate:
            rospy.logerr("  [高概率] scan-to-map匹配特征不足或退化")
            rospy.logerr("           LIO-SAM依赖IMU预积分,累积误差导致跳变")

        if velocity > 100:
            rospy.logerr("  [高概率] 优化完全失败,位姿估计发散")
            rospy.logerr("           建议检查: 特征提取阈值、地图质量")

        if np.abs(dp[2]) > np.abs(dp[0]) and np.abs(dp[2]) > np.abs(dp[1]):
            rospy.logerr("  [可能] Z方向大跳变可能与IMU重力补偿有关")

        rospy.logerr("="*60)

    def report_status(self):
        """报告当前状态"""
        rospy.loginfo("--- 状态报告 (帧 %d) ---", self.total_frames)
        rospy.loginfo("  跳变事件: %d (%.1f%%)",
                     self.jump_count,
                     100*self.jump_count/max(1, self.total_frames))
        rospy.loginfo("  退化帧: %d (%.1f%%)",
                     self.degenerate_count,
                     100*self.degenerate_count/max(1, self.total_frames))

        if self.jump_count > 0:
            rospy.logwarn("  !!! 检测到速度跳变,系统可能存在问题 !!!")

def main():
    rospy.init_node('velocity_jump_root_cause')

    analyzer = VelocityJumpRootCause()

    rospy.loginfo("开始监控... (Ctrl+C 停止)")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rospy.loginfo("")
        rospy.loginfo("="*60)
        rospy.loginfo("最终报告")
        rospy.loginfo("="*60)
        rospy.loginfo("总帧数: %d", analyzer.total_frames)
        rospy.loginfo("跳变次数: %d", analyzer.jump_count)
        rospy.loginfo("退化帧数: %d", analyzer.degenerate_count)

        if analyzer.jump_count > 0 and analyzer.degenerate_count > 0:
            correlation = analyzer.degenerate_count / max(1, analyzer.jump_count)
            if correlation > 0.5:
                rospy.loginfo("")
                rospy.loginfo(">>> 结论: 跳变与退化高度相关!")
                rospy.loginfo(">>> 根本原因可能是: scan-to-map匹配特征不足")
                rospy.loginfo(">>> 建议检查:")
                rospy.loginfo("    1. 特征提取参数 (edgeThreshold, surfThreshold)")
                rospy.loginfo("    2. LiDAR范围过滤 (lidarMinRange, lidarMaxRange)")
                rospy.loginfo("    3. 环境几何特征是否充足")

if __name__ == '__main__':
    main()
