#!/usr/bin/env python3
"""
综合数据正确性检查
"""

import rosbag
import numpy as np
from collections import defaultdict

bag_file = '/root/autodl-tmp/info_fixed.bag'

print("="*70)
print("综合数据正确性检查")
print("="*70)

bag = rosbag.Bag(bag_file, 'r')

# 1. IMU 数据分析
print("\n" + "="*70)
print("1. IMU 数据分析 (/imu/data)")
print("="*70)

imu_times = []
imu_data = []
for topic, msg, t in bag.read_messages(topics=['/imu/data']):
    imu_times.append(msg.header.stamp.to_sec())
    imu_data.append({
        'acc': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
        'gyro': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
        'quat': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    })
    if len(imu_times) >= 1000:
        break

if imu_times:
    dt_list = np.diff(imu_times)
    freq = 1.0 / np.mean(dt_list)

    print(f"采样数: {len(imu_times)}")
    print(f"时间范围: {imu_times[0]:.3f} - {imu_times[-1]:.3f}")
    print(f"频率: {freq:.1f} Hz (期望: 200 Hz)")
    print(f"平均dt: {np.mean(dt_list)*1000:.2f} ms")
    print(f"dt范围: [{np.min(dt_list)*1000:.2f}, {np.max(dt_list)*1000:.2f}] ms")

    # 加速度分析
    accs = np.array([d['acc'] for d in imu_data])
    acc_norm = np.linalg.norm(accs, axis=1)
    print(f"\n加速度:")
    print(f"  X: [{accs[:,0].min():.2f}, {accs[:,0].max():.2f}] m/s²")
    print(f"  Y: [{accs[:,1].min():.2f}, {accs[:,1].max():.2f}] m/s²")
    print(f"  Z: [{accs[:,2].min():.2f}, {accs[:,2].max():.2f}] m/s²")
    print(f"  范数: [{acc_norm.min():.2f}, {acc_norm.max():.2f}] m/s² (期望~9.8)")

    # 判断重力方向
    mean_acc = np.mean(accs, axis=0)
    print(f"  平均值: [{mean_acc[0]:.2f}, {mean_acc[1]:.2f}, {mean_acc[2]:.2f}]")
    gravity_axis = np.argmax(np.abs(mean_acc))
    axis_names = ['X', 'Y', 'Z']
    print(f"  重力主轴: {axis_names[gravity_axis]} (值: {mean_acc[gravity_axis]:.2f})")

    # 角速度分析
    gyros = np.array([d['gyro'] for d in imu_data])
    print(f"\n角速度:")
    print(f"  X: [{gyros[:,0].min():.3f}, {gyros[:,0].max():.3f}] rad/s")
    print(f"  Y: [{gyros[:,1].min():.3f}, {gyros[:,1].max():.3f}] rad/s")
    print(f"  Z: [{gyros[:,2].min():.3f}, {gyros[:,2].max():.3f}] rad/s")

    # 四元数分析
    quats = np.array([d['quat'] for d in imu_data])
    quat_norm = np.linalg.norm(quats, axis=1)
    print(f"\n四元数:")
    print(f"  范数: [{quat_norm.min():.4f}, {quat_norm.max():.4f}] (期望: 1.0)")
    if np.all(quats == 0):
        print(f"  ⚠️  警告: 四元数全为0!")

# 2. GPS/FPA Odometry 数据分析
print("\n" + "="*70)
print("2. FPA Odometry 数据分析")
print("="*70)

# 检查原始 FPA odometry
fpa_times = []
fpa_positions = []
for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
    fpa_times.append(msg.header.stamp.to_sec())
    fpa_positions.append([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ])
    if len(fpa_times) >= 100:
        break

if fpa_positions:
    pos = np.array(fpa_positions)
    print(f"\n/fixposition/fpa/odomenu (ENU坐标):")
    print(f"  采样数: {len(fpa_times)}")
    print(f"  X范围: [{pos[:,0].min():.2f}, {pos[:,0].max():.2f}] m")
    print(f"  Y范围: [{pos[:,1].min():.2f}, {pos[:,1].max():.2f}] m")
    print(f"  Z范围: [{pos[:,2].min():.2f}, {pos[:,2].max():.2f}] m")

    # 检查是否是合理的局部坐标
    if np.abs(pos).max() > 10000:
        print(f"  ⚠️  警告: 坐标值过大，可能不是ENU坐标!")
    else:
        print(f"  ✅ 坐标值在合理范围内")

# 检查原始 FPA odometry (ECEF)
fpa_ecef_times = []
fpa_ecef_positions = []
for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
    fpa_ecef_times.append(msg.header.stamp.to_sec())
    fpa_ecef_positions.append([
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        msg.pose.pose.position.z
    ])
    if len(fpa_ecef_times) >= 100:
        break

if fpa_ecef_positions:
    pos = np.array(fpa_ecef_positions)
    print(f"\n/fixposition/fpa/odometry (ECEF坐标):")
    print(f"  采样数: {len(fpa_ecef_times)}")
    print(f"  X范围: [{pos[:,0].min():.2f}, {pos[:,0].max():.2f}] m")
    print(f"  Y范围: [{pos[:,1].min():.2f}, {pos[:,1].max():.2f}] m")
    print(f"  Z范围: [{pos[:,2].min():.2f}, {pos[:,2].max():.2f}] m")
    print(f"  距地心距离: {np.linalg.norm(pos[0]):.0f} m (期望~6371km)")

# 3. 时间同步分析
print("\n" + "="*70)
print("3. 时间同步分析")
print("="*70)

# 获取各传感器第一个时间戳
first_times = {}
for topic in ['/lidar_points', '/imu/data', '/fixposition/fpa/odomenu']:
    for t_name, msg, t in bag.read_messages(topics=[topic]):
        first_times[topic] = msg.header.stamp.to_sec()
        break

if first_times:
    base_time = min(first_times.values())
    print(f"\n各传感器起始时间 (相对于最早时间):")
    for topic, time in sorted(first_times.items(), key=lambda x: x[1]):
        print(f"  {topic}: +{(time - base_time)*1000:.1f} ms")

    # 检查IMU是否比LiDAR早
    if '/imu/data' in first_times and '/lidar_points' in first_times:
        diff = first_times['/lidar_points'] - first_times['/imu/data']
        if diff > 0:
            print(f"\n  ✅ IMU比LiDAR早 {diff*1000:.1f} ms (正确)")
        else:
            print(f"\n  ⚠️  LiDAR比IMU早 {-diff*1000:.1f} ms (可能导致deskew失败)")

# 4. LiDAR 与 IMU 时间戳对齐检查
print("\n" + "="*70)
print("4. LiDAR-IMU 时间对齐检查")
print("="*70)

lidar_times = []
for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
    lidar_times.append(msg.header.stamp.to_sec())
    if len(lidar_times) >= 10:
        break

imu_times_full = []
for topic, msg, t in bag.read_messages(topics=['/imu/data']):
    imu_times_full.append(msg.header.stamp.to_sec())

if lidar_times and imu_times_full:
    imu_times_arr = np.array(imu_times_full)

    print(f"\n前10帧LiDAR时间戳与IMU覆盖情况:")
    for i, lt in enumerate(lidar_times[:5]):
        # 找到该LiDAR帧前后的IMU数据
        before = np.sum(imu_times_arr < lt)
        after = np.sum(imu_times_arr > lt)

        # 找最近的IMU时间
        if before > 0:
            nearest_before = imu_times_arr[imu_times_arr < lt].max()
            diff_before = (lt - nearest_before) * 1000
        else:
            diff_before = float('inf')

        if after > 0:
            nearest_after = imu_times_arr[imu_times_arr > lt].min()
            diff_after = (nearest_after - lt) * 1000
        else:
            diff_after = float('inf')

        status = "✅" if diff_before < 10 and diff_after < 10 else "⚠️"
        print(f"  Frame {i}: t={lt:.3f}, IMU before: {before}, after: {after}")
        print(f"    {status} 最近IMU: -{diff_before:.1f}ms / +{diff_after:.1f}ms")

# 5. 协方差检查
print("\n" + "="*70)
print("5. GPS协方差检查")
print("="*70)

cov_data = []
for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
    cov = msg.pose.covariance
    cov_data.append([cov[0], cov[7], cov[14]])  # xx, yy, zz
    if len(cov_data) >= 100:
        break

if cov_data:
    cov_arr = np.array(cov_data)
    print(f"\n位置协方差 (m²):")
    print(f"  X: [{cov_arr[:,0].min():.4f}, {cov_arr[:,0].max():.4f}]")
    print(f"  Y: [{cov_arr[:,1].min():.4f}, {cov_arr[:,1].max():.4f}]")
    print(f"  Z: [{cov_arr[:,2].min():.4f}, {cov_arr[:,2].max():.4f}]")

    # 检查是否会被gpsCovThreshold过滤
    gps_cov_threshold = 2.0
    filtered = np.sum((cov_arr[:,0] > gps_cov_threshold) | (cov_arr[:,1] > gps_cov_threshold))
    print(f"\n  被过滤的GPS点 (cov > {gps_cov_threshold}): {filtered}/{len(cov_arr)}")

bag.close()

print("\n" + "="*70)
print("检查完成")
print("="*70)
