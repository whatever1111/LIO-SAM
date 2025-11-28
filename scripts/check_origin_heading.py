#!/usr/bin/env python3
"""
诊断脚本：检查GPS和LIO-SAM的坐标系原点和航向初始化问题
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation
import sys

def quaternion_to_euler(qx, qy, qz, qw):
    """四元数转欧拉角 (roll, pitch, yaw)"""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz', degrees=True)

def analyze_origin_and_heading(bag_file):
    """分析GPS和IMU的原点和航向"""

    print("=" * 70)
    print("坐标系原点和航向初始化诊断")
    print("=" * 70)
    print(f"\nBag文件: {bag_file}\n")

    # 读取数据
    gps_data = []
    imu_data = []
    lidar_times = []

    bag = rosbag.Bag(bag_file, 'r')

    # 获取话题信息
    topics = bag.get_type_and_topic_info()[1]
    print("可用话题:")
    for topic, info in topics.items():
        print(f"  {topic}: {info.message_count} 条消息")
    print()

    # 读取GPS数据 (FPA Odometry)
    gps_topic = None
    for topic in ['/fixposition/fpa/odometry', '/odometry/gps']:
        if topic in topics:
            gps_topic = topic
            break

    if gps_topic:
        print(f"读取GPS数据: {gps_topic}")
        for topic, msg, t in bag.read_messages(topics=[gps_topic]):
            if hasattr(msg, 'pose'):
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                gps_data.append({
                    'time': msg.header.stamp.to_sec(),
                    'x': pos.x, 'y': pos.y, 'z': pos.z,
                    'qx': ori.x, 'qy': ori.y, 'qz': ori.z, 'qw': ori.w
                })
            if len(gps_data) >= 100:  # 只读取前100条
                break

    # 读取IMU数据
    imu_topic = '/imu/data'
    if imu_topic in topics:
        print(f"读取IMU数据: {imu_topic}")
        for topic, msg, t in bag.read_messages(topics=[imu_topic]):
            ori = msg.orientation
            imu_data.append({
                'time': msg.header.stamp.to_sec(),
                'qx': ori.x, 'qy': ori.y, 'qz': ori.z, 'qw': ori.w
            })
            if len(imu_data) >= 100:
                break

    # 读取LiDAR时间戳
    lidar_topic = '/lidar_points'
    if lidar_topic in topics:
        print(f"读取LiDAR时间戳: {lidar_topic}")
        for topic, msg, t in bag.read_messages(topics=[lidar_topic]):
            lidar_times.append(msg.header.stamp.to_sec())
            if len(lidar_times) >= 10:
                break

    bag.close()

    print("\n" + "=" * 70)
    print("1. 坐标系原点分析")
    print("=" * 70)

    if gps_data:
        first_gps = gps_data[0]
        print(f"\n[GPS原点] - 第一条GPS消息:")
        print(f"  时间戳: {first_gps['time']:.6f}")
        print(f"  位置 (ECEF/原始): x={first_gps['x']:.2f}, y={first_gps['y']:.2f}, z={first_gps['z']:.2f}")

        # 检查是否是ECEF坐标 (ECEF坐标通常很大，几百万米)
        if abs(first_gps['x']) > 1e6 or abs(first_gps['y']) > 1e6:
            print(f"  -> 这是ECEF坐标 (地心坐标系)")
            print(f"  -> fpaOdomConverter会将此设为ENU原点 (0, 0, 0)")
        else:
            print(f"  -> 这已经是局部坐标 (可能已经转换过)")
    else:
        print("  [警告] 未找到GPS数据!")

    if lidar_times:
        print(f"\n[LIO-SAM原点] - 第一帧LiDAR:")
        print(f"  时间戳: {lidar_times[0]:.6f}")
        print(f"  LIO-SAM会将第一帧位置设为 (0, 0, 0)")

    # 时间差分析
    if gps_data and lidar_times:
        time_diff = lidar_times[0] - gps_data[0]['time']
        print(f"\n[时间差分析]")
        print(f"  GPS第一条消息时间: {gps_data[0]['time']:.6f}")
        print(f"  LiDAR第一帧时间:   {lidar_times[0]:.6f}")
        print(f"  时间差: {time_diff*1000:.2f} ms")

        if abs(time_diff) > 0.1:
            print(f"\n  [问题] GPS和LiDAR的第一条数据时间差 > 100ms!")
            print(f"         这可能导致两个原点代表不同的物理位置")

            # 估算位置差异 (假设车速10m/s)
            estimated_drift = abs(time_diff) * 10  # 假设10m/s
            print(f"         假设车速10m/s，位置差异可能达到 {estimated_drift:.2f} 米")
        else:
            print(f"  [OK] 时间差在合理范围内")

    print("\n" + "=" * 70)
    print("2. 航向初始化分析")
    print("=" * 70)

    if imu_data:
        first_imu = imu_data[0]
        roll, pitch, yaw = quaternion_to_euler(
            first_imu['qx'], first_imu['qy'], first_imu['qz'], first_imu['qw']
        )
        print(f"\n[IMU初始航向] - 第一条IMU消息:")
        print(f"  时间戳: {first_imu['time']:.6f}")
        print(f"  四元数: ({first_imu['qw']:.4f}, {first_imu['qx']:.4f}, {first_imu['qy']:.4f}, {first_imu['qz']:.4f})")
        print(f"  欧拉角: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
        print(f"\n  -> LIO-SAM将使用此航向 (Yaw={yaw:.2f}°) 作为初始方向")

    if gps_data:
        first_gps = gps_data[0]
        gps_roll, gps_pitch, gps_yaw = quaternion_to_euler(
            first_gps['qx'], first_gps['qy'], first_gps['qz'], first_gps['qw']
        )
        print(f"\n[GPS初始航向] - 第一条GPS消息:")
        print(f"  四元数: ({first_gps['qw']:.4f}, {first_gps['qx']:.4f}, {first_gps['qy']:.4f}, {first_gps['qz']:.4f})")
        print(f"  欧拉角: Roll={gps_roll:.2f}°, Pitch={gps_pitch:.2f}°, Yaw={gps_yaw:.2f}°")

    if imu_data and gps_data:
        yaw_diff = yaw - gps_yaw
        # 归一化到 [-180, 180]
        while yaw_diff > 180:
            yaw_diff -= 360
        while yaw_diff < -180:
            yaw_diff += 360

        print(f"\n[航向差异分析]")
        print(f"  IMU航向: {yaw:.2f}°")
        print(f"  GPS航向: {gps_yaw:.2f}°")
        print(f"  差异: {yaw_diff:.2f}°")

        if abs(yaw_diff) > 5:
            print(f"\n  [问题] IMU和GPS航向差异 > 5°!")
            print(f"         这会导致LIO-SAM坐标系与GPS的ENU坐标系有旋转偏差")

            # 估算位置误差 (假设100米距离)
            distance = 100  # 米
            position_error = distance * np.tan(np.radians(abs(yaw_diff)))
            print(f"         在100米距离处，这会导致约 {position_error:.2f} 米的横向偏差")
        else:
            print(f"  [OK] 航向差异在合理范围内")

    print("\n" + "=" * 70)
    print("3. 坐标系定义分析")
    print("=" * 70)

    print("""
GPS坐标系 (ENU - 东-北-天):
  - X轴: 指向东 (East)
  - Y轴: 指向北 (North)
  - Z轴: 指向天 (Up)
  - 航向0°: 朝向北方

LIO-SAM坐标系 (使用IMU初始化):
  - X轴: 前方 (由IMU定义)
  - Y轴: 左方
  - Z轴: 上方
  - 航向0°: 由IMU初始航向决定

如果IMU初始时不朝向北方，两个坐标系就会存在旋转偏差！
""")

    print("=" * 70)
    print("4. 建议")
    print("=" * 70)

    suggestions = []

    if gps_data and lidar_times and abs(lidar_times[0] - gps_data[0]['time']) > 0.1:
        suggestions.append("""
[原点同步问题]
- 确保GPS和LiDAR数据在相近时间开始记录
- 或者在代码中显式对齐两个原点""")

    if imu_data and gps_data and abs(yaw_diff) > 5:
        suggestions.append(f"""
[航向对齐问题]
- 当前IMU和GPS航向差异: {yaw_diff:.2f}°
- 建议方案:
  1. 在params.yaml中设置 useImuHeadingInitialization: false
     (这会将初始yaw设为0)
  2. 或者在fpaOdomConverter中将GPS坐标系旋转到与IMU对齐
  3. 或者使用GPS航向来初始化LIO-SAM的航向""")

    if suggestions:
        for s in suggestions:
            print(s)
    else:
        print("\n未发现明显问题，但仍需结合实际运行结果判断。")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        bag_file = '/root/autodl-tmp/info_fixed.bag'
    else:
        bag_file = sys.argv[1]

    analyze_origin_and_heading(bag_file)
