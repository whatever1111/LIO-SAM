#!/usr/bin/env python3

import rospy
import rostopic
import time
import numpy as np
from collections import deque, defaultdict
import threading
import signal
import sys
from datetime import datetime

class LIOSAMPerformanceMonitor:
    def __init__(self):
        rospy.init_node('liosam_performance_monitor', anonymous=True)

        # 存储各个话题的时间戳和延迟
        self.timestamps = defaultdict(lambda: deque(maxlen=100))
        self.delays = defaultdict(lambda: deque(maxlen=100))
        self.message_counts = defaultdict(int)
        self.last_print_time = time.time()

        # 要监控的话题
        self.topics_to_monitor = [
            ('/livox/lidar', 'sensor_msgs/PointCloud2'),  # 原始点云输入
            ('/lio_sam/deskew/cloud_info', 'lio_sam/cloud_info'),  # imageProjection 输出
            ('/lio_sam/feature/cloud_info', 'lio_sam/cloud_info'),  # featureExtraction 输出
            ('/lio_sam/mapping/cloud_registered', 'sensor_msgs/PointCloud2'),  # mapOptmization 输出
            ('/odometry/imu_incremental', 'nav_msgs/Odometry'),  # IMU 预积分输出
            ('/lio_sam/mapping/odometry', 'nav_msgs/Odometry'),  # 最终里程计输出
        ]

        self.module_delays = {
            'imageProjection': deque(maxlen=100),
            'featureExtraction': deque(maxlen=100),
            'mapOptmization': deque(maxlen=100),
            'imuPreintegration': deque(maxlen=100),
            'total_pipeline': deque(maxlen=100)
        }

        # 启动监听器
        self.start_monitoring()

    def get_header_time(self, msg):
        """从消息中提取header时间戳"""
        try:
            if hasattr(msg, 'header'):
                return msg.header.stamp.to_sec()
            elif hasattr(msg, 'cloud_header'):
                return msg.cloud_header.stamp.to_sec()
        except:
            pass
        return None

    def callback_factory(self, topic_name):
        """为每个话题创建回调函数"""
        def callback(msg):
            receive_time = rospy.Time.now().to_sec()
            header_time = self.get_header_time(msg)

            if header_time:
                delay = (receive_time - header_time) * 1000  # 转换为毫秒
                self.delays[topic_name].append(delay)
                self.timestamps[topic_name].append(header_time)
                self.message_counts[topic_name] += 1

                # 计算模块间延迟
                self.calculate_module_delays()

        return callback

    def calculate_module_delays(self):
        """计算各个模块之间的处理延迟"""
        try:
            # 获取最新的时间戳
            if len(self.timestamps['/livox/lidar']) > 0 and len(self.timestamps['/lio_sam/deskew/cloud_info']) > 0:
                # imageProjection 处理时间
                latest_input = self.timestamps['/livox/lidar'][-1]
                latest_deskew = self.timestamps['/lio_sam/deskew/cloud_info'][-1]
                if abs(latest_deskew - latest_input) < 1.0:  # 同一帧
                    self.module_delays['imageProjection'].append((latest_deskew - latest_input) * 1000)

            if len(self.timestamps['/lio_sam/deskew/cloud_info']) > 0 and len(self.timestamps['/lio_sam/feature/cloud_info']) > 0:
                # featureExtraction 处理时间
                latest_deskew = self.timestamps['/lio_sam/deskew/cloud_info'][-1]
                latest_feature = self.timestamps['/lio_sam/feature/cloud_info'][-1]
                if abs(latest_feature - latest_deskew) < 1.0:
                    self.module_delays['featureExtraction'].append((latest_feature - latest_deskew) * 1000)

            if len(self.timestamps['/livox/lidar']) > 0 and len(self.timestamps['/lio_sam/mapping/odometry']) > 0:
                # 总体延迟
                latest_input = self.timestamps['/livox/lidar'][-1]
                latest_output = self.timestamps['/lio_sam/mapping/odometry'][-1]
                if abs(latest_output - latest_input) < 2.0:
                    self.module_delays['total_pipeline'].append((latest_output - latest_input) * 1000)

        except Exception as e:
            pass

    def start_monitoring(self):
        """启动话题监听"""
        self.subscribers = []
        for topic, msg_type in self.topics_to_monitor:
            try:
                # 动态导入消息类型
                msg_class = rostopic.get_topic_class(topic)[0]
                if msg_class:
                    sub = rospy.Subscriber(topic, msg_class, self.callback_factory(topic))
                    self.subscribers.append(sub)
                    print(f"✓ 监听话题: {topic}")
                else:
                    print(f"✗ 无法找到话题: {topic}")
            except Exception as e:
                print(f"✗ 监听话题 {topic} 失败: {e}")

    def print_statistics(self):
        """打印性能统计信息"""
        current_time = time.time()
        if current_time - self.last_print_time < 2.0:  # 每2秒打印一次
            return

        self.last_print_time = current_time

        print("\n" + "="*80)
        print(f"LIO-SAM 性能监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # 打印话题延迟统计
        print("\n📊 话题延迟统计（ms）:")
        print("-"*60)
        print(f"{'话题名称':<40} {'平均':<8} {'最小':<8} {'最大':<8} {'消息数':<8}")
        print("-"*60)

        for topic, _ in self.topics_to_monitor:
            if len(self.delays[topic]) > 0:
                delays = list(self.delays[topic])
                avg_delay = np.mean(delays)
                min_delay = np.min(delays)
                max_delay = np.max(delays)
                count = self.message_counts[topic]

                # 根据延迟大小着色
                color = '\033[92m' if avg_delay < 100 else '\033[93m' if avg_delay < 200 else '\033[91m'
                reset = '\033[0m'

                topic_short = topic.split('/')[-2] + '/' + topic.split('/')[-1] if '/' in topic else topic
                print(f"{topic_short:<40} {color}{avg_delay:>7.1f}{reset} {min_delay:>7.1f} {max_delay:>7.1f} {count:>7}")

        # 打印模块处理时间统计
        print("\n⏱️ 模块处理时间（ms）:")
        print("-"*60)
        print(f"{'模块名称':<25} {'平均':<10} {'最小':<10} {'最大':<10} {'P95':<10}")
        print("-"*60)

        for module, delays in self.module_delays.items():
            if len(delays) > 0:
                delays_list = list(delays)
                avg_time = np.mean(delays_list)
                min_time = np.min(delays_list)
                max_time = np.max(delays_list)
                p95_time = np.percentile(delays_list, 95)

                # 根据时间大小着色
                color = '\033[92m' if avg_time < 50 else '\033[93m' if avg_time < 100 else '\033[91m'
                reset = '\033[0m'

                print(f"{module:<25} {color}{avg_time:>9.1f}{reset} {min_time:>9.1f} {max_time:>9.1f} {p95_time:>9.1f}")

        # 打印警告信息
        if len(self.module_delays['total_pipeline']) > 0:
            total_delays = list(self.module_delays['total_pipeline'])
            avg_total = np.mean(total_delays)
            max_total = np.max(total_delays)

            print("\n⚠️ 性能警告:")
            if avg_total > 200:
                print(f"  • 平均总延迟 {avg_total:.0f}ms 超过 200ms 阈值!")
            if max_total > 1000:
                print(f"  • 最大延迟 {max_total:.0f}ms 超过 1000ms!")

            # 找出瓶颈模块
            bottleneck = max(self.module_delays.items(),
                           key=lambda x: np.mean(list(x[1])) if len(x[1]) > 0 else 0)
            if len(bottleneck[1]) > 0:
                print(f"  • 主要瓶颈: {bottleneck[0]} (平均 {np.mean(list(bottleneck[1])):.0f}ms)")

    def run(self):
        """主运行循环"""
        rate = rospy.Rate(1)  # 1Hz
        while not rospy.is_shutdown():
            self.print_statistics()
            rate.sleep()

def signal_handler(sig, frame):
    print("\n\n🛑 停止性能监控...")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    monitor = LIOSAMPerformanceMonitor()

    print("\n🚀 LIO-SAM 性能监控已启动!")
    print("   按 Ctrl+C 停止监控\n")

    try:
        monitor.run()
    except rospy.ROSInterruptException:
        pass