#!/usr/bin/env python3
"""
深入分析GPS四元数和坐标系问题
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation
import sys

def quaternion_to_euler_xyz(qx, qy, qz, qw):
    """四元数转欧拉角 (xyz顺序)"""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz', degrees=True)

def quaternion_to_euler_zyx(qx, qy, qz, qw):
    """四元数转欧拉角 (zyx顺序 - 常用于航空)"""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_euler('zyx', degrees=True)

def analyze_gps_orientation(bag_file):
    """深入分析GPS四元数"""

    print("=" * 70)
    print("GPS四元数和坐标系深入分析")
    print("=" * 70)

    bag = rosbag.Bag(bag_file, 'r')

    # 读取GPS和IMU的前几条数据
    gps_data = []
    imu_data = []

    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        gps_data.append({
            'time': msg.header.stamp.to_sec(),
            'x': pos.x, 'y': pos.y, 'z': pos.z,
            'qx': ori.x, 'qy': ori.y, 'qz': ori.z, 'qw': ori.w,
            'frame': msg.header.frame_id,
            'child_frame': msg.child_frame_id if hasattr(msg, 'child_frame_id') else 'N/A'
        })
        if len(gps_data) >= 20:
            break

    for topic, msg, t in bag.read_messages(topics=['/imu/data']):
        ori = msg.orientation
        imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'qx': ori.x, 'qy': ori.y, 'qz': ori.z, 'qw': ori.w,
            'frame': msg.header.frame_id
        })
        if len(imu_data) >= 20:
            break

    # 也读取 odomenu 话题看看有什么不同
    odomenu_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
        if hasattr(msg, 'pose'):
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            odomenu_data.append({
                'time': msg.header.stamp.to_sec(),
                'x': pos.x, 'y': pos.y, 'z': pos.z,
                'qx': ori.x, 'qy': ori.y, 'qz': ori.z, 'qw': ori.w,
                'frame': msg.header.frame_id
            })
        if len(odomenu_data) >= 5:
            break

    bag.close()

    print("\n" + "=" * 70)
    print("1. GPS (/fixposition/fpa/odometry) 四元数分析")
    print("=" * 70)

    if gps_data:
        g = gps_data[0]
        print(f"\n第一条消息 frame_id: {g['frame']}")
        print(f"child_frame_id: {g['child_frame']}")
        print(f"\n四元数 (qx, qy, qz, qw): ({g['qx']:.6f}, {g['qy']:.6f}, {g['qz']:.6f}, {g['qw']:.6f})")

        # 检查四元数是否归一化
        norm = np.sqrt(g['qx']**2 + g['qy']**2 + g['qz']**2 + g['qw']**2)
        print(f"四元数范数: {norm:.6f} (应该接近1.0)")

        # 不同顺序的欧拉角
        euler_xyz = quaternion_to_euler_xyz(g['qx'], g['qy'], g['qz'], g['qw'])
        euler_zyx = quaternion_to_euler_zyx(g['qx'], g['qy'], g['qz'], g['qw'])

        print(f"\n欧拉角 (xyz顺序): Roll={euler_xyz[0]:.2f}°, Pitch={euler_xyz[1]:.2f}°, Yaw={euler_xyz[2]:.2f}°")
        print(f"欧拉角 (zyx顺序): Yaw={euler_zyx[0]:.2f}°, Pitch={euler_zyx[1]:.2f}°, Roll={euler_zyx[2]:.2f}°")

        print("\n前5条GPS数据的位置变化:")
        for i, g in enumerate(gps_data[:5]):
            euler = quaternion_to_euler_xyz(g['qx'], g['qy'], g['qz'], g['qw'])
            print(f"  [{i}] t={g['time']:.3f} pos=({g['x']:.1f}, {g['y']:.1f}, {g['z']:.1f}) "
                  f"euler=({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)")

    print("\n" + "=" * 70)
    print("2. OdomENU (/fixposition/fpa/odomenu) 分析")
    print("=" * 70)

    if odomenu_data:
        print("\n注意: odomenu 话题可能已经是ENU坐标系")
        for i, g in enumerate(odomenu_data[:3]):
            euler = quaternion_to_euler_xyz(g['qx'], g['qy'], g['qz'], g['qw'])
            print(f"  [{i}] t={g['time']:.3f} pos=({g['x']:.2f}, {g['y']:.2f}, {g['z']:.2f}) "
                  f"euler=({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)")
            print(f"       frame: {g['frame']}")
    else:
        print("未找到 odomenu 数据")

    print("\n" + "=" * 70)
    print("3. IMU 四元数分析")
    print("=" * 70)

    if imu_data:
        i = imu_data[0]
        print(f"\n第一条消息 frame_id: {i['frame']}")
        print(f"四元数 (qx, qy, qz, qw): ({i['qx']:.6f}, {i['qy']:.6f}, {i['qz']:.6f}, {i['qw']:.6f})")

        euler_xyz = quaternion_to_euler_xyz(i['qx'], i['qy'], i['qz'], i['qw'])
        euler_zyx = quaternion_to_euler_zyx(i['qx'], i['qy'], i['qz'], i['qw'])

        print(f"\n欧拉角 (xyz顺序): Roll={euler_xyz[0]:.2f}°, Pitch={euler_xyz[1]:.2f}°, Yaw={euler_xyz[2]:.2f}°")
        print(f"欧拉角 (zyx顺序): Yaw={euler_zyx[0]:.2f}°, Pitch={euler_zyx[1]:.2f}°, Roll={euler_zyx[2]:.2f}°")

    print("\n" + "=" * 70)
    print("4. 关键发现")
    print("=" * 70)

    if gps_data and imu_data:
        g = gps_data[0]
        i = imu_data[0]

        gps_euler = quaternion_to_euler_xyz(g['qx'], g['qy'], g['qz'], g['qw'])
        imu_euler = quaternion_to_euler_xyz(i['qx'], i['qy'], i['qz'], i['qw'])

        print(f"""
[问题1] GPS四元数可能使用不同的坐标系约定:
  - /fixposition/fpa/odometry 的四元数可能是ECEF坐标系下的姿态
  - 而不是ENU坐标系下的姿态
  - Roll={gps_euler[0]:.1f}° 表明姿态定义与通常的ENU不同

[问题2] 坐标系转换不完整:
  - fpaOdomConverter只转换了位置 (ECEF -> ENU)
  - 但直接复制了四元数，没有旋转姿态:
    nav_odom.pose.pose.orientation = fpa_msg->pose.pose.orientation;

[问题3] 应该使用 odomenu 话题:
  - /fixposition/fpa/odomenu 可能已经是ENU坐标系
  - 不需要再做ECEF->ENU转换
""")

    print("\n" + "=" * 70)
    print("5. 建议解决方案")
    print("=" * 70)

    print("""
方案A: 使用 /fixposition/fpa/odomenu 话题
  - 这个话题可能已经是ENU坐标系，不需要再转换
  - 修改 fpaOdomConverter 直接使用 odomenu 数据

方案B: 修复四元数转换
  - 在 fpaOdomConverter 中，将ECEF姿态转换为ENU姿态
  - 需要用 R_ecef_to_enu 旋转四元数

方案C: 对齐LIO-SAM和GPS的航向
  - 修改 mapOptmization.cpp，使用GPS的初始航向
  - 或者在初始化时校正航向偏差
""")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        bag_file = '/root/autodl-tmp/info_fixed.bag'
    else:
        bag_file = sys.argv[1]

    analyze_gps_orientation(bag_file)
