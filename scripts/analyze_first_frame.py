#!/usr/bin/env python3
"""
分析第一帧的位姿初始化
"""

import rosbag
import numpy as np

bag_file = '/root/autodl-tmp/info_fixed.bag'

print("="*70)
print("第一帧位姿初始化分析")
print("="*70)

bag = rosbag.Bag(bag_file, 'r')

# 获取第一个IMU数据的姿态
print("\n1. 第一个IMU数据:")
for topic, msg, t in bag.read_messages(topics=['/imu/data']):
    q = msg.orientation
    print(f"   时间: {msg.header.stamp.to_sec():.6f}")
    print(f"   四元数: [{q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f}]")

    # 转换为欧拉角
    import tf.transformations as tf_trans
    euler = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
    roll, pitch, yaw = np.degrees(euler)
    print(f"   欧拉角: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")
    break

# 获取第一个LiDAR数据
print("\n2. 第一个LiDAR数据:")
for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
    print(f"   时间: {msg.header.stamp.to_sec():.6f}")
    print(f"   点数: {msg.width * msg.height}")
    break

# 检查LiDAR和IMU时间差
print("\n3. 分析IMU在第一帧时的姿态:")
lidar_first_time = None
for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
    lidar_first_time = msg.header.stamp.to_sec()
    break

# 找到最接近第一帧LiDAR时间的IMU数据
closest_imu = None
min_diff = float('inf')
for topic, msg, t in bag.read_messages(topics=['/imu/data']):
    diff = abs(msg.header.stamp.to_sec() - lidar_first_time)
    if diff < min_diff:
        min_diff = diff
        closest_imu = msg
    if msg.header.stamp.to_sec() > lidar_first_time + 0.1:
        break

if closest_imu:
    q = closest_imu.orientation
    euler = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
    roll, pitch, yaw = np.degrees(euler)
    print(f"   最近IMU时间差: {min_diff*1000:.1f} ms")
    print(f"   欧拉角: roll={roll:.2f}°, pitch={pitch:.2f}°, yaw={yaw:.2f}°")

    # 这个roll/pitch会导致的位置偏移 (如果有高度差)
    # 例如,如果传感器离地面1米,roll=5度会导致约0.087米的水平偏移
    print("\n4. 初始姿态对位置的影响:")
    print(f"   如果传感器高度为1m:")
    roll_rad, pitch_rad = np.radians(roll), np.radians(pitch)
    dx_from_roll = 1.0 * np.sin(roll_rad)
    dy_from_pitch = 1.0 * np.sin(pitch_rad)
    print(f"   roll导致的X偏移: {dx_from_roll:.3f} m")
    print(f"   pitch导致的Y偏移: {dy_from_pitch:.3f} m")

bag.close()

print("\n" + "="*70)
