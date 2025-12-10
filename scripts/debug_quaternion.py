#!/usr/bin/env python3
"""
调试四元数积分 - 直接打印原始数据对比
"""

import rospy
import math
import numpy as np
from sensor_msgs.msg import Imu
from collections import deque

try:
    from fixposition_driver_msgs.msg import FpaImu
    HAS_FPA = True
except:
    HAS_FPA = False


def quat_to_euler(qx, qy, qz, qw):
    """四元数转欧拉角"""
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1, min(1, sinp)))

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


class QuaternionDebugger:
    def __init__(self):
        rospy.init_node('quaternion_debugger', anonymous=True)

        self.corrimu_data = deque(maxlen=50)
        self.lio_imu_data = deque(maxlen=50)

        if HAS_FPA:
            rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.corrimu_cb)
        rospy.Subscriber('/lio_sam/imu/data', Imu, self.lio_imu_cb)

        rospy.Timer(rospy.Duration(2.0), self.analyze)

        print("=" * 80)
        print("四元数积分调试工具")
        print("=" * 80)

    def corrimu_cb(self, msg):
        self.corrimu_data.append({
            'time': msg.data.header.stamp.to_sec(),
            'gyr_x': msg.data.angular_velocity.x,
            'gyr_y': msg.data.angular_velocity.y,
            'gyr_z': msg.data.angular_velocity.z,
        })

    def lio_imu_cb(self, msg):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)

        self.lio_imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'gyr_x': msg.angular_velocity.x,
            'gyr_y': msg.angular_velocity.y,
            'gyr_z': msg.angular_velocity.z,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
        })

    def analyze(self, event):
        if len(self.lio_imu_data) < 10:
            print(f"等待数据... LIO-IMU: {len(self.lio_imu_data)}")
            return

        print("\n" + "=" * 80)

        # 1. 打印最近几帧的数据
        print("\n[最近5帧 LIO-SAM IMU 数据]")
        print(f"{'时间':>12} | {'gyr_x':>8} {'gyr_y':>8} {'gyr_z':>8} | {'Roll':>8} {'Pitch':>8} {'Yaw':>8}")
        print("-" * 80)

        data = list(self.lio_imu_data)[-5:]
        for d in data:
            t = d['time'] % 1000  # 只显示秒的小数部分
            print(f"{t:12.3f} | {d['gyr_x']:8.4f} {d['gyr_y']:8.4f} {d['gyr_z']:8.4f} | "
                  f"{math.degrees(d['roll']):8.2f} {math.degrees(d['pitch']):8.2f} {math.degrees(d['yaw']):8.2f}")

        # 2. 手动积分验证
        print("\n[手动积分验证]")
        if len(self.lio_imu_data) >= 20:
            data = list(self.lio_imu_data)

            # 取最近20帧，手动用角速度积分yaw
            yaw_integrated = data[0]['yaw']
            for i in range(1, min(20, len(data))):
                dt = data[i]['time'] - data[i-1]['time']
                if 0 < dt < 0.1:
                    # 简单积分: yaw_new = yaw_old + gyr_z * dt
                    yaw_integrated += data[i]['gyr_z'] * dt

            yaw_actual = data[min(19, len(data)-1)]['yaw']

            print(f"  起始Yaw: {math.degrees(data[0]['yaw']):.2f}°")
            print(f"  手动积分Yaw (用gyr_z): {math.degrees(yaw_integrated):.2f}°")
            print(f"  实际Yaw: {math.degrees(yaw_actual):.2f}°")
            print(f"  差异: {math.degrees(yaw_actual - yaw_integrated):.2f}°")

            # 计算平均角速度
            gyr_z_mean = np.mean([d['gyr_z'] for d in data[:20]])
            time_span = data[min(19, len(data)-1)]['time'] - data[0]['time']
            expected_yaw_change = gyr_z_mean * time_span
            actual_yaw_change = yaw_actual - data[0]['yaw']

            print(f"\n  时间跨度: {time_span:.3f}s")
            print(f"  平均gyr_z: {gyr_z_mean:.4f} rad/s")
            print(f"  期望Yaw变化 (gyr_z*t): {math.degrees(expected_yaw_change):.2f}°")
            print(f"  实际Yaw变化: {math.degrees(actual_yaw_change):.2f}°")

        # 3. 对比CORRIMU原始角速度和LIO-SAM发布的角速度
        if len(self.corrimu_data) >= 10 and len(self.lio_imu_data) >= 10:
            print("\n[CORRIMU vs LIO-SAM 角速度对比]")
            corr = list(self.corrimu_data)[-10:]
            lio = list(self.lio_imu_data)[-10:]

            corr_gyr = np.array([[d['gyr_x'], d['gyr_y'], d['gyr_z']] for d in corr])
            lio_gyr = np.array([[d['gyr_x'], d['gyr_y'], d['gyr_z']] for d in lio])

            print(f"  CORRIMU 均值: gyr=({np.mean(corr_gyr[:,0]):.4f}, {np.mean(corr_gyr[:,1]):.4f}, {np.mean(corr_gyr[:,2]):.4f})")
            print(f"  LIO-SAM 均值: gyr=({np.mean(lio_gyr[:,0]):.4f}, {np.mean(lio_gyr[:,1]):.4f}, {np.mean(lio_gyr[:,2]):.4f})")

            # 检查变换关系 (extRot = Rz(180°) => gyr_x'=-gyr_x, gyr_y'=-gyr_y, gyr_z'=gyr_z)
            print(f"\n  预期变换 Rz(180°): gyr_x'=-gyr_x, gyr_y'=-gyr_y, gyr_z'=gyr_z")
            print(f"  CORRIMU gyr_x={np.mean(corr_gyr[:,0]):.4f} => LIO gyr_x={np.mean(lio_gyr[:,0]):.4f} (预期: {-np.mean(corr_gyr[:,0]):.4f})")
            print(f"  CORRIMU gyr_y={np.mean(corr_gyr[:,1]):.4f} => LIO gyr_y={np.mean(lio_gyr[:,1]):.4f} (预期: {-np.mean(corr_gyr[:,1]):.4f})")
            print(f"  CORRIMU gyr_z={np.mean(corr_gyr[:,2]):.4f} => LIO gyr_z={np.mean(lio_gyr[:,2]):.4f} (预期: {np.mean(corr_gyr[:,2]):.4f})")


def main():
    debugger = QuaternionDebugger()
    rospy.spin()


if __name__ == '__main__':
    main()
