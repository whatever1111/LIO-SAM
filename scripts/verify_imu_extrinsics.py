#!/usr/bin/env python3
"""
验证IMU外参脚本
分别检测 /fixposition/fpa/corrimu 和 /imu/data 的坐标系
通过比较加速度方向和GPS速度方向来判断正确的外参
"""

import rospy
import math
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from collections import deque

try:
    from fixposition_driver_msgs.msg import FpaImu
    HAS_FPA = True
except:
    HAS_FPA = False
    print("Warning: fixposition_driver_msgs not found, skipping FpaImu")

class IMUExtrinsicsVerifier:
    def __init__(self):
        rospy.init_node('imu_extrinsics_verifier', anonymous=True)

        # 数据缓存
        self.corrimu_data = deque(maxlen=100)
        self.imu_data = deque(maxlen=100)
        self.gps_data = deque(maxlen=100)

        self.last_gps_pos = None
        self.last_gps_time = None

        # 订阅
        if HAS_FPA:
            rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.corrimu_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb)

        # 定时分析
        rospy.Timer(rospy.Duration(3.0), self.analyze)

        print("=" * 70)
        print("IMU外参验证工具")
        print("=" * 70)
        print("订阅话题:")
        print("  - /fixposition/fpa/corrimu (CORRIMU)")
        print("  - /imu/data (标准IMU)")
        print("  - /odometry/gps (GPS)")
        print()
        print("分析方法: 比较IMU加速度方向和GPS速度变化方向")
        print("=" * 70)

    def corrimu_cb(self, msg):
        self.corrimu_data.append({
            'time': msg.data.header.stamp.to_sec(),
            'acc_x': msg.data.linear_acceleration.x,
            'acc_y': msg.data.linear_acceleration.y,
            'acc_z': msg.data.linear_acceleration.z,
            'gyr_x': msg.data.angular_velocity.x,
            'gyr_y': msg.data.angular_velocity.y,
            'gyr_z': msg.data.angular_velocity.z,
        })

    def imu_cb(self, msg):
        # 提取四元数的yaw
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)

        self.imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc_x': msg.linear_acceleration.x,
            'acc_y': msg.linear_acceleration.y,
            'acc_z': msg.linear_acceleration.z,
            'gyr_x': msg.angular_velocity.x,
            'gyr_y': msg.angular_velocity.y,
            'gyr_z': msg.angular_velocity.z,
            'yaw': yaw,
        })

    def gps_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 提取四元数yaw
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny, cosy)

        # 计算速度
        vx, vy = 0, 0
        if self.last_gps_pos is not None:
            dt = t - self.last_gps_time
            if dt > 0.01:
                vx = (x - self.last_gps_pos[0]) / dt
                vy = (y - self.last_gps_pos[1]) / dt

        self.last_gps_pos = (x, y)
        self.last_gps_time = t

        self.gps_data.append({
            'time': t,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'yaw': yaw,
        })

    def analyze(self, event):
        print("\n" + "=" * 70)
        print(f"数据统计: CORRIMU={len(self.corrimu_data)}, IMU={len(self.imu_data)}, GPS={len(self.gps_data)}")
        print("=" * 70)

        # 1. 分析CORRIMU
        if len(self.corrimu_data) > 10:
            self.analyze_corrimu()

        # 2. 分析/imu/data
        if len(self.imu_data) > 10:
            self.analyze_imu_data()

        # 3. 分析GPS
        if len(self.gps_data) > 10:
            self.analyze_gps()

        # 4. 外参建议
        self.suggest_extrinsics()

    def analyze_corrimu(self):
        print("\n[CORRIMU 分析]")
        acc_x = np.mean([d['acc_x'] for d in self.corrimu_data])
        acc_y = np.mean([d['acc_y'] for d in self.corrimu_data])
        acc_z = np.mean([d['acc_z'] for d in self.corrimu_data])

        print(f"  平均加速度: X={acc_x:.3f}, Y={acc_y:.3f}, Z={acc_z:.3f} m/s²")

        # 重力方向检测
        acc_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        print(f"  加速度模长: {acc_mag:.3f} m/s² (应≈9.8)")

        if abs(acc_z) > 8:
            print(f"  重力方向: Z轴 ({'向下' if acc_z > 0 else '向上'})")
        elif abs(acc_x) > 8:
            print(f"  重力方向: X轴 (异常!)")
        elif abs(acc_y) > 8:
            print(f"  重力方向: Y轴 (异常!)")

    def analyze_imu_data(self):
        print("\n[/imu/data 分析]")
        acc_x = np.mean([d['acc_x'] for d in self.imu_data])
        acc_y = np.mean([d['acc_y'] for d in self.imu_data])
        acc_z = np.mean([d['acc_z'] for d in self.imu_data])
        yaw_mean = np.mean([d['yaw'] for d in self.imu_data])

        print(f"  平均加速度: X={acc_x:.3f}, Y={acc_y:.3f}, Z={acc_z:.3f} m/s²")
        print(f"  平均航向: {math.degrees(yaw_mean):.1f}°")

        acc_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        print(f"  加速度模长: {acc_mag:.3f} m/s² (应≈9.8)")

    def analyze_gps(self):
        print("\n[GPS 分析]")
        # 取最近的有效速度数据
        valid_gps = [d for d in self.gps_data if math.sqrt(d['vx']**2 + d['vy']**2) > 0.5]

        if valid_gps:
            vx_mean = np.mean([d['vx'] for d in valid_gps])
            vy_mean = np.mean([d['vy'] for d in valid_gps])
            yaw_mean = np.mean([d['yaw'] for d in valid_gps])

            speed = math.sqrt(vx_mean**2 + vy_mean**2)
            vel_heading = math.degrees(math.atan2(vy_mean, vx_mean))

            print(f"  平均速度: Vx={vx_mean:.2f}, Vy={vy_mean:.2f} m/s (速度={speed:.2f} m/s)")
            print(f"  速度航向: {vel_heading:.1f}°")
            print(f"  四元数航向: {math.degrees(yaw_mean):.1f}°")
            print(f"  航向差(四元数-速度): {math.degrees(yaw_mean) - vel_heading:.1f}°")
        else:
            print("  速度数据不足")

    def suggest_extrinsics(self):
        print("\n" + "=" * 70)
        print("[外参建议]")
        print("=" * 70)

        if len(self.imu_data) > 10 and len(self.gps_data) > 10:
            # 比较/imu/data的航向和GPS航向
            imu_yaw = math.degrees(np.mean([d['yaw'] for d in self.imu_data]))
            gps_yaws = [d['yaw'] for d in self.gps_data if d['yaw'] != 0]
            if gps_yaws:
                gps_yaw = math.degrees(np.mean(gps_yaws))
                yaw_diff = imu_yaw - gps_yaw
                # 归一化到[-180, 180]
                while yaw_diff > 180: yaw_diff -= 360
                while yaw_diff < -180: yaw_diff += 360

                print(f"\n/imu/data航向: {imu_yaw:.1f}°")
                print(f"GPS航向: {gps_yaw:.1f}°")
                print(f"差值: {yaw_diff:.1f}°")

                if abs(yaw_diff) < 15:
                    print("\n建议: /imu/data 航向与GPS基本一致")
                    print("  extrinsicRPY 使用 Rz(180°) = [-1,0,0; 0,-1,0; 0,0,1]")
                elif abs(yaw_diff - 90) < 15 or abs(yaw_diff + 270) < 15:
                    print("\n建议: /imu/data 航向比GPS多90°")
                    print("  extrinsicRPY 使用 Rz(180°-90°) = Rz(90°) = [0,-1,0; 1,0,0; 0,0,1]")
                elif abs(yaw_diff + 90) < 15 or abs(yaw_diff - 270) < 15:
                    print("\n建议: /imu/data 航向比GPS少90°")
                    print("  extrinsicRPY 使用 Rz(180°+90°) = Rz(-90°) = [0,1,0; -1,0,0; 0,0,1]")
                elif abs(abs(yaw_diff) - 180) < 15:
                    print("\n建议: /imu/data 航向与GPS相差180°")
                    print("  extrinsicRPY 使用 Rz(0°) = [1,0,0; 0,1,0; 0,0,1]")
                else:
                    print(f"\n航向差异 {yaw_diff:.1f}° 不是典型值，请检查传感器安装")

        print("\n注意: CORRIMU没有姿态数据，初始航向默认为0")
        print("      如果使用CORRIMU，需要依赖GPS Factor来校正航向")

def main():
    verifier = IMUExtrinsicsVerifier()
    rospy.spin()

if __name__ == '__main__':
    main()
