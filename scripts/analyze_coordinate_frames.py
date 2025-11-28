#!/usr/bin/env python3
"""
分析GPS ENU坐标系和LIO-SAM坐标系是否对齐
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation
import sys

def analyze_coordinate_frames(bag_file):
    """分析坐标系定义"""

    print("=" * 70)
    print("坐标系对齐分析")
    print("=" * 70)

    bag = rosbag.Bag(bag_file, 'r')

    # 读取数据
    gps_odom = []      # /fixposition/fpa/odometry (ECEF)
    gps_enu = []       # /fixposition/fpa/odomenu (ENU)
    imu_data = []

    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        gps_odom.append({
            'time': msg.header.stamp.to_sec(),
            'pos': np.array([pos.x, pos.y, pos.z]),
            'quat': np.array([ori.x, ori.y, ori.z, ori.w])
        })
        if len(gps_odom) >= 50:
            break

    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        gps_enu.append({
            'time': msg.header.stamp.to_sec(),
            'pos': np.array([pos.x, pos.y, pos.z]),
            'quat': np.array([ori.x, ori.y, ori.z, ori.w])
        })
        if len(gps_enu) >= 50:
            break

    for topic, msg, t in bag.read_messages(topics=['/imu/data']):
        ori = msg.orientation
        imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'quat': np.array([ori.x, ori.y, ori.z, ori.w])
        })
        if len(imu_data) >= 50:
            break

    bag.close()

    print("\n" + "=" * 70)
    print("1. GPS ENU (odomenu) 坐标系分析")
    print("=" * 70)

    if len(gps_enu) > 10:
        # 计算运动方向
        pos_start = gps_enu[0]['pos']
        pos_end = gps_enu[-1]['pos']
        delta_pos = pos_end - pos_start

        print(f"\n位置变化:")
        print(f"  起始: ({pos_start[0]:.3f}, {pos_start[1]:.3f}, {pos_start[2]:.3f})")
        print(f"  结束: ({pos_end[0]:.3f}, {pos_end[1]:.3f}, {pos_end[2]:.3f})")
        print(f"  位移: dx={delta_pos[0]:.3f}, dy={delta_pos[1]:.3f}, dz={delta_pos[2]:.3f}")

        # 从位移计算运动方向角
        motion_heading_enu = np.arctan2(delta_pos[0], delta_pos[1]) * 180 / np.pi  # ENU中Y是北
        print(f"  运动方向角 (ENU): {motion_heading_enu:.2f}° (从北向顺时针)")

        # GPS航向
        r = Rotation.from_quat(gps_enu[0]['quat'])
        euler = r.as_euler('zyx', degrees=True)  # yaw, pitch, roll
        print(f"\nGPS姿态 (ENU odomenu):")
        print(f"  四元数: {gps_enu[0]['quat']}")
        print(f"  Yaw={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Roll={euler[2]:.2f}°")

    print("\n" + "=" * 70)
    print("2. IMU 坐标系分析")
    print("=" * 70)

    if imu_data:
        r = Rotation.from_quat(imu_data[0]['quat'])
        euler = r.as_euler('zyx', degrees=True)
        print(f"\nIMU初始姿态:")
        print(f"  四元数: {imu_data[0]['quat']}")
        print(f"  Yaw={euler[0]:.2f}°, Pitch={euler[1]:.2f}°, Roll={euler[2]:.2f}°")

    print("\n" + "=" * 70)
    print("3. 坐标系定义对比")
    print("=" * 70)

    print("""
标准ENU坐标系 (GPS使用):
  - X: 指向东 (East)
  - Y: 指向北 (North)
  - Z: 指向天 (Up)
  - Yaw=0°: 指向北, Yaw=90°: 指向东

ROS REP-105 (LIO-SAM使用):
  - X: 指向前 (Forward)
  - Y: 指向左 (Left)
  - Z: 指向上 (Up)
  - Yaw=0°: 指向前进方向
""")

    if len(gps_enu) > 10 and imu_data:
        gps_yaw = Rotation.from_quat(gps_enu[0]['quat']).as_euler('zyx', degrees=True)[0]
        imu_yaw = Rotation.from_quat(imu_data[0]['quat']).as_euler('zyx', degrees=True)[0]

        print(f"\n实际测量:")
        print(f"  GPS ENU Yaw: {gps_yaw:.2f}°")
        print(f"  IMU Yaw:     {imu_yaw:.2f}°")
        print(f"  差异:        {imu_yaw - gps_yaw:.2f}°")

        # 分析ENU到Body的转换
        print(f"\n" + "=" * 70)
        print("4. 关键分析")
        print("=" * 70)

        # ENU中，如果Yaw=-5.8°，表示车辆几乎朝北(Y轴正方向)稍微偏西
        # IMU中，Yaw=84.2°，表示车辆朝向与IMU X轴正方向成84.2°角

        print(f"""
分析GPS ENU数据:
  - GPS Yaw = {gps_yaw:.2f}°
  - 在ENU中，Yaw=0°表示朝北(Y轴正方向)
  - GPS Yaw = {gps_yaw:.2f}° 表示车辆几乎朝北，稍微偏{"西" if gps_yaw < 0 else "东"}

分析IMU数据:
  - IMU Yaw = {imu_yaw:.2f}°
  - IMU的Yaw定义取决于IMU坐标系的定义
  - 如果IMU X轴朝前，Yaw={imu_yaw:.2f}°表示车辆朝向与参考方向成{imu_yaw:.2f}°角

关键问题:
  1. GPS使用ENU坐标系，其中Yaw是相对于北方向的角度
  2. LIO-SAM使用IMU初始航向作为Yaw=0的参考
  3. 如果两者的Yaw=0参考不同，坐标系就不对齐！

差异 = {imu_yaw - gps_yaw:.2f}°
这意味着LIO-SAM坐标系相对于GPS ENU坐标系旋转了约{imu_yaw - gps_yaw:.2f}°
""")

        # 验证：检查运动方向
        if abs(delta_pos[0]) > 0.1 or abs(delta_pos[1]) > 0.1:
            print(f"\n运动方向验证:")
            print(f"  GPS ENU位移: dx={delta_pos[0]:.3f}m (东), dy={delta_pos[1]:.3f}m (北)")

            # 如果车辆主要朝北移动，dy应该大于dx
            if abs(delta_pos[1]) > abs(delta_pos[0]):
                print(f"  -> 车辆主要向北移动")
            else:
                print(f"  -> 车辆主要向东移动")

            print(f"\n  如果LIO-SAM认为车辆向前移动(X方向)，")
            print(f"  但GPS显示车辆向北移动(Y方向)，")
            print(f"  则两个坐标系的X-Y轴定义不同！")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        bag_file = '/root/autodl-tmp/info_fixed.bag'
    else:
        bag_file = sys.argv[1]

    analyze_coordinate_frames(bag_file)
