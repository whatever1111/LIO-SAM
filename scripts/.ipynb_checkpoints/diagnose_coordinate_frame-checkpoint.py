#!/usr/bin/env python3
"""
精确诊断GPS与LIO-SAM坐标系对齐问题
通过累积位移对比来确定正确的gpsExtrinsicRot
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from collections import deque
import threading

class CoordinateFrameDiagnosis:
    def __init__(self):
        rospy.init_node('coordinate_frame_diagnosis', anonymous=True)

        self.gps_positions = []
        self.fusion_positions = []
        self.lock = threading.Lock()
        self.start_time = None

        rospy.Subscriber("/odometry/gps", Odometry, self.gps_callback)
        rospy.Subscriber("/lio_sam/mapping/odometry", Odometry, self.fusion_callback)

        print("=" * 80)
        print("坐标系对齐诊断工具")
        print("=" * 80)
        print("请让车辆直线行驶至少10米后查看结果")
        print("-" * 80)

    def gps_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            self.gps_positions.append({
                'time': t,
                'rel_time': t - self.start_time,
                'x': msg.pose.pose.position.x,  # ENU East
                'y': msg.pose.pose.position.y,  # ENU North
                'z': msg.pose.pose.position.z   # ENU Up
            })

    def fusion_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            if self.start_time is None:
                self.start_time = t

            self.fusion_positions.append({
                'time': t,
                'rel_time': t - self.start_time,
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            })

    def analyze(self):
        with self.lock:
            if len(self.gps_positions) < 10 or len(self.fusion_positions) < 10:
                print("数据不足，请等待...")
                return

            # 计算GPS累积位移
            gps_start = self.gps_positions[0]
            gps_end = self.gps_positions[-1]
            gps_dx = gps_end['x'] - gps_start['x']  # East
            gps_dy = gps_end['y'] - gps_start['y']  # North
            gps_dz = gps_end['z'] - gps_start['z']  # Up
            gps_dist = np.sqrt(gps_dx**2 + gps_dy**2)

            # 计算融合累积位移
            fus_start = self.fusion_positions[0]
            fus_end = self.fusion_positions[-1]
            fus_dx = fus_end['x'] - fus_start['x']
            fus_dy = fus_end['y'] - fus_start['y']
            fus_dz = fus_end['z'] - fus_start['z']
            fus_dist = np.sqrt(fus_dx**2 + fus_dy**2)

            time_span = max(gps_end['rel_time'], fus_end['rel_time'])

            print("\n" + "=" * 80)
            print(f"坐标系对齐诊断 (数据时长: {time_span:.1f}s)")
            print("=" * 80)

            print(f"\n【GPS ENU累积位移】")
            print(f"  ΔX (East):  {gps_dx:+8.2f} m")
            print(f"  ΔY (North): {gps_dy:+8.2f} m")
            print(f"  ΔZ (Up):    {gps_dz:+8.2f} m")
            print(f"  水平距离:   {gps_dist:8.2f} m")

            print(f"\n【融合轨迹累积位移】")
            print(f"  ΔX:         {fus_dx:+8.2f} m")
            print(f"  ΔY:         {fus_dy:+8.2f} m")
            print(f"  ΔZ:         {fus_dz:+8.2f} m")
            print(f"  水平距离:   {fus_dist:8.2f} m")

            # 分析对应关系
            print(f"\n【坐标轴对应关系分析】")

            # 检查各种可能的映射: 将GPS坐标转换后与融合比较
            # 格式: (名称, 转换后X, 转换后Y)
            mappings = [
                ("GPS直接使用 (X=E,Y=N)",     gps_dx,  gps_dy),      # Identity
                ("GPS Y反向 (X=E,Y=-N)",      gps_dx, -gps_dy),      # Y flip
                ("GPS X反向 (X=-E,Y=N)",     -gps_dx,  gps_dy),      # X flip
                ("GPS XY反向 (X=-E,Y=-N)",   -gps_dx, -gps_dy),      # 180 rotation
                ("GPS XY交换 (X=N,Y=E)",      gps_dy,  gps_dx),      # swap
                ("GPS XY交换+Y反 (X=N,Y=-E)", gps_dy, -gps_dx),      # 90 CW
                ("GPS XY交换+X反 (X=-N,Y=E)",-gps_dy,  gps_dx),      # 90 CCW
                ("GPS XY交换+双反 (X=-N,Y=-E)",-gps_dy,-gps_dx),     # swap + 180
            ]

            best_mapping = None
            best_error = float('inf')
            best_transformed = None

            print(f"\n  映射关系                      | GPS变换后       |  融合实际        | 误差")
            print(f"  " + "-" * 75)

            for name, trans_x, trans_y in mappings:
                error = np.sqrt((trans_x - fus_dx)**2 + (trans_y - fus_dy)**2)

                mark = ""
                if error < best_error:
                    best_error = error
                    best_mapping = name
                    best_transformed = (trans_x, trans_y)
                    mark = " ← 最佳"

                print(f"  {name:30s} | ({trans_x:+6.1f},{trans_y:+6.1f}) | ({fus_dx:+6.1f},{fus_dy:+6.1f}) | {error:5.1f}m{mark}")

            print(f"\n【推荐的gpsExtrinsicRot配置】")

            if best_mapping:
                print(f"  最佳匹配: {best_mapping} (误差: {best_error:.1f}m)")

                # 根据最佳映射推荐gpsExtrinsicRot
                rot_configs = {
                    "GPS直接使用 (X=E,Y=N)":     ("[ 1,  0,  0,\n                      0,  1,  0,\n                      0,  0,  1]", "无旋转"),
                    "GPS Y反向 (X=E,Y=-N)":      ("[ 1,  0,  0,\n                      0, -1,  0,\n                      0,  0,  1]", "Y轴反向"),
                    "GPS X反向 (X=-E,Y=N)":      ("[-1,  0,  0,\n                      0,  1,  0,\n                      0,  0,  1]", "X轴反向"),
                    "GPS XY反向 (X=-E,Y=-N)":    ("[-1,  0,  0,\n                      0, -1,  0,\n                      0,  0,  1]", "180度旋转"),
                    "GPS XY交换 (X=N,Y=E)":      ("[ 0,  1,  0,\n                      1,  0,  0,\n                      0,  0,  1]", "XY交换"),
                    "GPS XY交换+Y反 (X=N,Y=-E)": ("[ 0,  1,  0,\n                     -1,  0,  0,\n                      0,  0,  1]", "90度顺时针"),
                    "GPS XY交换+X反 (X=-N,Y=E)": ("[ 0, -1,  0,\n                      1,  0,  0,\n                      0,  0,  1]", "90度逆时针"),
                    "GPS XY交换+双反 (X=-N,Y=-E)":("[ 0, -1,  0,\n                     -1,  0,  0,\n                      0,  0,  1]", "XY交换+180度"),
                }

                if best_mapping in rot_configs:
                    rot, desc = rot_configs[best_mapping]
                    print(f"\n  gpsExtrinsicRot: {rot}")
                    print(f"  含义: {desc}")
                else:
                    print(f"\n  无法自动推荐配置，请手动计算")

            # 额外检查：如果位移太小，警告用户
            if gps_dist < 5 or fus_dist < 5:
                print(f"\n  [警告] 位移太小 (GPS:{gps_dist:.1f}m, 融合:{fus_dist:.1f}m)")
                print(f"         请让车辆直线行驶更长距离以获得准确诊断")

            # 检查距离比例
            if gps_dist > 1 and fus_dist > 1:
                ratio = fus_dist / gps_dist
                print(f"\n  距离比例: 融合/GPS = {ratio:.2f}")
                if abs(ratio - 1.0) > 0.3:
                    print(f"  [警告] 距离比例偏离较大，可能存在尺度问题")

            print("\n" + "=" * 80)

    def run(self):
        rate = rospy.Rate(0.5)
        try:
            while not rospy.is_shutdown():
                self.analyze()
                rate.sleep()
        except KeyboardInterrupt:
            print("\n最终诊断:")
            self.analyze()

if __name__ == '__main__':
    try:
        diag = CoordinateFrameDiagnosis()
        diag.run()
    except rospy.ROSInterruptException:
        pass
