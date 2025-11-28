#!/usr/bin/env python3
"""
测试延迟监控功能
生成模拟数据来验证 realtime_plotter.py 的延迟图表
"""

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import numpy as np
import time

def test_latency_monitor():
    """生成测试数据来验证延迟监控"""

    rospy.init_node('latency_test_publisher')

    # 发布器
    fusion_pub = rospy.Publisher('/lio_sam/mapping/odometry', Odometry, queue_size=10)
    gps_pub = rospy.Publisher('/odometry/gps', Odometry, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz

    rospy.loginfo("Starting latency test publisher...")
    rospy.loginfo("This will publish simulated odometry with varying latencies")

    t = 0
    while not rospy.is_shutdown():
        # 创建消息
        fusion_msg = Odometry()
        gps_msg = Odometry()

        # 模拟不同的延迟场景
        if t < 100:
            # 正常延迟 (20-30ms)
            latency_offset = 0.025
        elif t < 200:
            # 中等延迟 (50-70ms)
            latency_offset = 0.060
        else:
            # 高延迟 (100-150ms)
            latency_offset = 0.125

        # 添加一些随机变化
        latency_offset += np.random.normal(0, 0.01)

        # 设置时间戳（带延迟）
        current_time = rospy.Time.now()
        delayed_time = current_time - rospy.Duration.from_sec(latency_offset)

        # 填充消息
        fusion_msg.header.stamp = delayed_time
        fusion_msg.header.frame_id = "odom"
        fusion_msg.pose.pose.position.x = t * 0.1
        fusion_msg.pose.pose.position.y = np.sin(t * 0.1) * 5
        fusion_msg.pose.pose.position.z = 0

        gps_msg.header.stamp = delayed_time
        gps_msg.header.frame_id = "odom"
        gps_msg.pose.pose.position.x = t * 0.1 + np.random.normal(0, 0.1)
        gps_msg.pose.pose.position.y = np.sin(t * 0.1) * 5 + np.random.normal(0, 0.1)
        gps_msg.pose.pose.position.z = 0

        # 发布
        fusion_pub.publish(fusion_msg)
        gps_pub.publish(gps_msg)

        if t % 50 == 0:
            rospy.loginfo(f"Time: {t/10:.1f}s, Simulated latency: {latency_offset*1000:.1f}ms")

        t += 1
        rate.sleep()

if __name__ == '__main__':
    try:
        test_latency_monitor()
    except rospy.ROSInterruptException:
        pass