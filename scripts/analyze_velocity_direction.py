#!/usr/bin/env python3
"""
分析GPS、IMU和融合轨迹的速度方向是否一致
用于诊断坐标系对齐问题

检测方法:
1. 比较GPS ENU速度方向与融合轨迹速度方向
2. 检查是否存在180度反向
3. 检查X/Y轴是否交换
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from collections import deque
import threading
import math

class VelocityDirectionAnalyzer:
    def __init__(self):
        rospy.init_node('velocity_direction_analyzer', anonymous=True)

        # 数据缓存
        self.gps_data = deque(maxlen=500)
        self.imu_data = deque(maxlen=500)
        self.fusion_data = deque(maxlen=500)

        # 位置历史用于计算速度
        self.gps_pos_history = deque(maxlen=10)
        self.fusion_pos_history = deque(maxlen=10)

        # 速度方向统计
        self.direction_comparisons = []
        self.angle_diffs = []

        # 锁
        self.lock = threading.Lock()

        # 订阅话题
        # GPS odometry (ENU坐标)
        rospy.Subscriber("/odometry/gps", Odometry, self.gps_callback)
        # IMU预积分 odometry
        rospy.Subscriber("/odometry/imu_incremental", Odometry, self.imu_callback)
        # 融合后的odometry
        rospy.Subscriber("/lio_sam/mapping/odometry", Odometry, self.fusion_callback)

        self.start_time = None

        print("=" * 80)
        print("GPS/IMU/Fusion 速度方向一致性分析器")
        print("=" * 80)
        print()
        print("用途: 检测坐标系是否对齐")
        print("  - 如果GPS和融合轨迹速度方向相反 → 坐标系可能有180度偏差")
        print("  - 如果X/Y方向交换 → gpsExtrinsicRot配置可能有误")
        print()
        print("订阅话题:")
        print("  - /odometry/gps (GPS ENU)")
        print("  - /odometry/imu_incremental (IMU预积分)")
        print("  - /lio_sam/mapping/odometry (融合轨迹)")
        print("-" * 80)

    def gps_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            # GPS位置 (ENU)
            pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            # GPS速度 (如果有的话)
            vel = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])

            self.gps_pos_history.append({'time': t, 'pos': pos})

            # 计算位置导数作为速度（如果twist为0）
            vel_from_pos = None
            if len(self.gps_pos_history) >= 2:
                p1 = self.gps_pos_history[-2]
                p2 = self.gps_pos_history[-1]
                dt = p2['time'] - p1['time']
                if dt > 0.01:
                    vel_from_pos = (p2['pos'] - p1['pos']) / dt

            self.gps_data.append({
                'time': t,
                'rel_time': t - self.start_time,
                'pos': pos,
                'vel': vel,
                'vel_from_pos': vel_from_pos
            })

    def imu_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

            vel = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])

            self.imu_data.append({
                'time': t,
                'rel_time': t - self.start_time if self.start_time else 0,
                'pos': pos,
                'vel': vel
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

            self.fusion_pos_history.append({'time': t, 'pos': pos})

            # 计算速度
            vel_from_pos = None
            if len(self.fusion_pos_history) >= 2:
                p1 = self.fusion_pos_history[-2]
                p2 = self.fusion_pos_history[-1]
                dt = p2['time'] - p1['time']
                if dt > 0.01:
                    vel_from_pos = (p2['pos'] - p1['pos']) / dt

            self.fusion_data.append({
                'time': t,
                'rel_time': t - self.start_time if self.start_time else 0,
                'pos': pos,
                'vel_from_pos': vel_from_pos
            })

    def find_nearest(self, data_list, target_time, max_diff=0.1):
        """找到最接近目标时间的数据"""
        best = None
        best_diff = float('inf')
        for item in data_list:
            diff = abs(item['time'] - target_time)
            if diff < best_diff and diff < max_diff:
                best_diff = diff
                best = item
        return best

    def angle_between_vectors(self, v1, v2):
        """计算两个向量之间的夹角（度）"""
        # 只考虑XY平面
        v1_2d = v1[:2]
        v2_2d = v2[:2]

        norm1 = np.linalg.norm(v1_2d)
        norm2 = np.linalg.norm(v2_2d)

        if norm1 < 0.1 or norm2 < 0.1:  # 速度太小，无法判断方向
            return None

        cos_angle = np.dot(v1_2d, v2_2d) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi

        return angle

    def get_heading(self, vel):
        """获取速度的航向角（度，从北开始顺时针）"""
        # 假设vel是ENU坐标系：X=East, Y=North
        vx, vy = vel[0], vel[1]
        if np.sqrt(vx**2 + vy**2) < 0.1:
            return None
        heading = np.arctan2(vx, vy) * 180 / np.pi  # atan2(E, N) 给出从北顺时针的角度
        return heading

    def analyze(self):
        """分析速度方向一致性"""
        with self.lock:
            if len(self.gps_data) < 5 or len(self.fusion_data) < 5:
                return

            print("\n" + "=" * 80)
            print(f"速度方向分析 (GPS: {len(self.gps_data)} samples, Fusion: {len(self.fusion_data)} samples)")
            print("=" * 80)

            comparisons = []

            # 遍历融合数据，找对应的GPS数据
            for fusion in list(self.fusion_data)[-100:]:  # 最近100个
                if fusion['vel_from_pos'] is None:
                    continue

                gps = self.find_nearest(self.gps_data, fusion['time'])
                if gps is None:
                    continue

                # 使用GPS位置计算的速度
                gps_vel = gps['vel_from_pos'] if gps['vel_from_pos'] is not None else gps['vel']
                if gps_vel is None:
                    continue

                fusion_vel = fusion['vel_from_pos']

                # 计算夹角
                angle = self.angle_between_vectors(gps_vel, fusion_vel)
                if angle is not None:
                    gps_heading = self.get_heading(gps_vel)
                    fusion_heading = self.get_heading(fusion_vel)

                    comparisons.append({
                        'time': fusion['rel_time'],
                        'gps_vel': gps_vel,
                        'fusion_vel': fusion_vel,
                        'angle_diff': angle,
                        'gps_heading': gps_heading,
                        'fusion_heading': fusion_heading
                    })

            if len(comparisons) < 3:
                print("数据不足，无法分析")
                return

            # 统计分析
            angles = [c['angle_diff'] for c in comparisons]
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            print(f"\n速度方向夹角统计:")
            print(f"  平均夹角: {mean_angle:.1f}° ± {std_angle:.1f}°")
            print(f"  最小夹角: {min(angles):.1f}°")
            print(f"  最大夹角: {max(angles):.1f}°")

            # 判断问题
            print(f"\n诊断结果:")
            if mean_angle < 30:
                print("  [OK] GPS和融合轨迹速度方向基本一致")
            elif 150 < mean_angle < 210 or mean_angle > 150:
                print("  [ERROR] GPS和融合轨迹速度方向接近相反!")
                print("         可能原因: gpsExtrinsicRot配置导致坐标系翻转")
            elif 60 < mean_angle < 120:
                print("  [WARNING] GPS和融合轨迹速度方向接近垂直!")
                print("            可能原因: X/Y轴交换或90度旋转错误")
            else:
                print(f"  [WARNING] 速度方向存在{mean_angle:.0f}度偏差")

            # 打印最近几个对比
            print(f"\n最近的对比 (时间 | GPS速度 | 融合速度 | 夹角):")
            print("-" * 80)
            for c in comparisons[-10:]:
                gps_v = c['gps_vel']
                fus_v = c['fusion_vel']
                gps_spd = np.linalg.norm(gps_v[:2])
                fus_spd = np.linalg.norm(fus_v[:2])

                gps_h = c['gps_heading']
                fus_h = c['fusion_heading']

                gps_h_str = f"{gps_h:+6.1f}°" if gps_h is not None else "  N/A "
                fus_h_str = f"{fus_h:+6.1f}°" if fus_h is not None else "  N/A "

                print(f"t={c['time']:6.1f}s | GPS:[{gps_v[0]:+6.2f},{gps_v[1]:+6.2f}] {gps_spd:4.1f}m/s {gps_h_str} | "
                      f"Fus:[{fus_v[0]:+6.2f},{fus_v[1]:+6.2f}] {fus_spd:4.1f}m/s {fus_h_str} | "
                      f"Δ={c['angle_diff']:5.1f}°")

            # 检查是否有系统性偏差
            if len(comparisons) > 10:
                recent = comparisons[-20:]
                gps_headings = [c['gps_heading'] for c in recent if c['gps_heading'] is not None]
                fusion_headings = [c['fusion_heading'] for c in recent if c['fusion_heading'] is not None]

                if len(gps_headings) > 5 and len(fusion_headings) > 5:
                    mean_gps_h = np.mean(gps_headings)
                    mean_fusion_h = np.mean(fusion_headings)
                    heading_diff = mean_fusion_h - mean_gps_h

                    # 归一化到 -180 到 180
                    while heading_diff > 180:
                        heading_diff -= 360
                    while heading_diff < -180:
                        heading_diff += 360

                    print(f"\n航向分析:")
                    print(f"  GPS平均航向: {mean_gps_h:+.1f}° (从北顺时针)")
                    print(f"  融合平均航向: {mean_fusion_h:+.1f}° (从北顺时针)")
                    print(f"  航向差: {heading_diff:+.1f}°")

                    if abs(heading_diff) > 150:
                        print(f"\n  [CRITICAL] 检测到约180°航向反转!")
                        print(f"             建议检查 gpsExtrinsicRot 配置")
                        print(f"             当前配置可能导致前后颠倒")
                    elif abs(heading_diff - 90) < 30 or abs(heading_diff + 90) < 30:
                        print(f"\n  [WARNING] 检测到约90°航向偏差!")
                        print(f"            可能是X/Y轴定义不一致")

            # 检查gpsExtrinsicRot的影响
            # 当前配置: [1,0,0; 0,-1,0; 0,0,1] (Y轴反向)
            print(f"\n当前gpsExtrinsicRot配置 (from params.yaml):")
            print(f"  配置: [ 1,  0,  0]   (X_out = X_enu = East)")
            print(f"        [ 0, -1,  0]   (Y_out = -Y_enu = South)")
            print(f"        [ 0,  0,  1]   (Z_out = Z_enu = Up)")
            print(f"  含义: GPS Y轴(北)反向为LIO-SAM的-Y(南)")

            # 显示原始位置变化，并应用gpsExtrinsicRot计算转换后的GPS
            if len(self.gps_data) > 5 and len(self.fusion_data) > 5:
                gps_first = list(self.gps_data)[0]
                gps_last = list(self.gps_data)[-1]
                fus_first = list(self.fusion_data)[0]
                fus_last = list(self.fusion_data)[-1]

                gps_delta = gps_last['pos'] - gps_first['pos']
                fus_delta = fus_last['pos'] - fus_first['pos']

                # 应用 gpsExtrinsicRot: [1,0,0; 0,-1,0; 0,0,1]
                gps_transformed = np.array([gps_delta[0], -gps_delta[1], gps_delta[2]])

                print(f"\n原始位置变化:")
                print(f"  GPS (ENU): ΔX={gps_delta[0]:+.2f}m(东), ΔY={gps_delta[1]:+.2f}m(北), ΔZ={gps_delta[2]:+.2f}m")
                print(f"  GPS转换后: ΔX={gps_transformed[0]:+.2f}m,      ΔY={gps_transformed[1]:+.2f}m,      ΔZ={gps_transformed[2]:+.2f}m")
                print(f"  融合:      ΔX={fus_delta[0]:+.2f}m,      ΔY={fus_delta[1]:+.2f}m,      ΔZ={fus_delta[2]:+.2f}m")

                # 计算转换后的误差
                error = np.linalg.norm(gps_transformed[:2] - fus_delta[:2])
                print(f"  转换后误差: {error:.2f}m")

                print(f"  时间跨度:  GPS={gps_last['rel_time']:.1f}s, 融合={fus_last['rel_time']:.1f}s")

            print("\n" + "=" * 80)

    def run(self):
        rate = rospy.Rate(0.5)  # 每2秒分析一次
        try:
            while not rospy.is_shutdown():
                self.analyze()
                rate.sleep()
        except KeyboardInterrupt:
            print("\n最终分析:")
            self.analyze()

if __name__ == '__main__':
    try:
        analyzer = VelocityDirectionAnalyzer()
        analyzer.run()
    except rospy.ROSInterruptException:
        pass
