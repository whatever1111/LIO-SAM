#!/usr/bin/env python3
"""
监控GPS与融合轨迹的长期对齐情况
检测100秒后的Y方向偏差问题
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import threading
from collections import deque

class LongTermAlignmentMonitor:
    def __init__(self):
        rospy.init_node('long_term_alignment_monitor', anonymous=True)

        self.gps_history = []
        self.fusion_history = []
        self.lock = threading.Lock()
        self.start_time = None

        # gpsExtrinsicRot配置: [1,0,0; 0,-1,0; 0,0,1]
        self.gps_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])

        rospy.Subscriber("/odometry/gps", Odometry, self.gps_callback)
        rospy.Subscriber("/lio_sam/mapping/odometry", Odometry, self.fusion_callback)

        print("=" * 80)
        print("GPS-Fusion长期对齐监控")
        print("=" * 80)
        print("监控GPS转换后与融合轨迹的累积误差")
        print("重点关注100秒后的Y方向漂移")
        print("-" * 80)

    def gps_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            # 应用gpsExtrinsicRot转换
            pos_transformed = self.gps_rot @ pos

            self.gps_history.append({
                'time': t,
                'rel_time': t - self.start_time,
                'pos_raw': pos,
                'pos_transformed': pos_transformed
            })

    def fusion_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            self.fusion_history.append({
                'time': t,
                'rel_time': t - self.start_time,
                'pos': pos
            })

    def find_nearest(self, history, target_time, max_diff=0.5):
        best = None
        best_diff = float('inf')
        for item in history:
            diff = abs(item['time'] - target_time)
            if diff < best_diff and diff < max_diff:
                best_diff = diff
                best = item
        return best

    def analyze(self):
        with self.lock:
            if len(self.gps_history) < 10 or len(self.fusion_history) < 10:
                print("数据收集中...")
                return

            # 使用第一个点作为原点
            gps_origin = self.gps_history[0]['pos_transformed']
            fus_origin = self.fusion_history[0]['pos']

            print("\n" + "=" * 80)
            print(f"长期对齐分析 (GPS: {len(self.gps_history)} pts, Fusion: {len(self.fusion_history)} pts)")
            print("=" * 80)

            # 按时间段分析误差
            time_ranges = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]

            print(f"\n时间段误差分析 (相对于起点):")
            print(f"{'时间段':>12} | {'GPS转换后(X,Y)':>20} | {'融合(X,Y)':>20} | {'X误差':>8} | {'Y误差':>8} | {'总误差':>8}")
            print("-" * 90)

            for t_start, t_end in time_ranges:
                # 找该时间段内的最后一个点
                gps_pts = [g for g in self.gps_history if t_start <= g['rel_time'] < t_end]
                fus_pts = [f for f in self.fusion_history if t_start <= f['rel_time'] < t_end]

                if not gps_pts or not fus_pts:
                    continue

                gps_last = gps_pts[-1]
                fus_last = fus_pts[-1]

                # 相对于原点的位移
                gps_delta = gps_last['pos_transformed'] - gps_origin
                fus_delta = fus_last['pos'] - fus_origin

                error_x = fus_delta[0] - gps_delta[0]
                error_y = fus_delta[1] - gps_delta[1]
                error_total = np.sqrt(error_x**2 + error_y**2)

                print(f"{t_start:3d}-{t_end:3d}s    | ({gps_delta[0]:+7.2f},{gps_delta[1]:+7.2f})  | ({fus_delta[0]:+7.2f},{fus_delta[1]:+7.2f})  | {error_x:+7.2f}m | {error_y:+7.2f}m | {error_total:7.2f}m")

            # 详细分析最近的数据
            print(f"\n最近10秒的详细对比:")
            print(f"{'时间':>8} | {'GPS转换后(X,Y,Z)':>25} | {'融合(X,Y,Z)':>25} | {'误差(X,Y)':>15}")
            print("-" * 85)

            recent_fus = [f for f in self.fusion_history if f['rel_time'] > self.fusion_history[-1]['rel_time'] - 10]

            for fus in recent_fus[-10:]:
                gps = self.find_nearest(self.gps_history, fus['time'])
                if gps is None:
                    continue

                gps_d = gps['pos_transformed'] - gps_origin
                fus_d = fus['pos'] - fus_origin

                err_x = fus_d[0] - gps_d[0]
                err_y = fus_d[1] - gps_d[1]

                print(f"t={fus['rel_time']:6.1f}s | ({gps_d[0]:+7.2f},{gps_d[1]:+7.2f},{gps_d[2]:+6.2f}) | ({fus_d[0]:+7.2f},{fus_d[1]:+7.2f},{fus_d[2]:+6.2f}) | ({err_x:+6.2f},{err_y:+6.2f})")

            # 分析Y方向漂移趋势
            if len(self.fusion_history) > 100:
                y_errors = []
                for fus in self.fusion_history[::10]:  # 每10个点取一个
                    gps = self.find_nearest(self.gps_history, fus['time'])
                    if gps:
                        gps_d = gps['pos_transformed'] - gps_origin
                        fus_d = fus['pos'] - fus_origin
                        y_errors.append({
                            'time': fus['rel_time'],
                            'error': fus_d[1] - gps_d[1]
                        })

                if len(y_errors) > 5:
                    times = [e['time'] for e in y_errors]
                    errors = [e['error'] for e in y_errors]

                    # 计算漂移速率
                    if max(times) > 10:
                        drift_rate = (errors[-1] - errors[0]) / (times[-1] - times[0]) if times[-1] > times[0] else 0
                        print(f"\nY方向漂移分析:")
                        print(f"  起始误差: {errors[0]:+.2f}m")
                        print(f"  当前误差: {errors[-1]:+.2f}m")
                        print(f"  漂移速率: {drift_rate:+.3f} m/s ({drift_rate*60:+.2f} m/min)")

                        if abs(drift_rate) > 0.05:
                            print(f"  [WARNING] Y方向漂移速率过高!")
                            print(f"            可能原因:")
                            print(f"            1. GPS factor没有有效融合 (检查poseCovThreshold)")
                            print(f"            2. 坐标系转换仍有问题")
                            print(f"            3. LiDAR里程计累积误差")

            print("\n" + "=" * 80)

    def run(self):
        rate = rospy.Rate(0.2)  # 每5秒分析一次
        try:
            while not rospy.is_shutdown():
                self.analyze()
                rate.sleep()
        except KeyboardInterrupt:
            print("\n最终分析:")
            self.analyze()

if __name__ == '__main__':
    try:
        monitor = LongTermAlignmentMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
