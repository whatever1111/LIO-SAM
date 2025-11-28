#!/usr/bin/env python3
"""
分析 LiDAR 和 IMU 时间戳差异的稳定性
"""

import rosbag
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bag_file = '/root/autodl-tmp/info_fixed.bag'

print("="*70)
print("LiDAR-IMU 时间戳差异稳定性分析")
print("="*70)

bag = rosbag.Bag(bag_file, 'r')

# 收集所有时间戳
lidar_times = []
imu_times = []

print("\n读取数据...")
for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
    lidar_times.append(msg.header.stamp.to_sec())

for topic, msg, t in bag.read_messages(topics=['/imu/data']):
    imu_times.append(msg.header.stamp.to_sec())

bag.close()

lidar_times = np.array(lidar_times)
imu_times = np.array(imu_times)

print(f"LiDAR帧数: {len(lidar_times)}")
print(f"IMU帧数: {len(imu_times)}")

# 分析每个LiDAR帧对应的IMU时间差
print("\n" + "="*70)
print("时间差分析")
print("="*70)

nearest_imu_before = []
nearest_imu_after = []
imu_count_in_window = []

for i, lt in enumerate(lidar_times):
    # 找到该LiDAR帧前后的IMU
    before_mask = imu_times < lt
    after_mask = imu_times > lt

    if np.any(before_mask):
        diff_before = lt - imu_times[before_mask].max()
        nearest_imu_before.append(diff_before * 1000)  # ms
    else:
        nearest_imu_before.append(np.nan)

    if np.any(after_mask):
        diff_after = imu_times[after_mask].min() - lt
        nearest_imu_after.append(diff_after * 1000)  # ms
    else:
        nearest_imu_after.append(np.nan)

    # 统计100ms窗口内的IMU数量
    window_mask = (imu_times >= lt - 0.05) & (imu_times <= lt + 0.05)
    imu_count_in_window.append(np.sum(window_mask))

nearest_imu_before = np.array(nearest_imu_before)
nearest_imu_after = np.array(nearest_imu_after)
imu_count_in_window = np.array(imu_count_in_window)

# 跳过第一帧(没有前面的IMU)
valid_before = nearest_imu_before[~np.isnan(nearest_imu_before)]
valid_after = nearest_imu_after[~np.isnan(nearest_imu_after)]

print(f"\n距离前一个IMU的时间差 (ms):")
print(f"  Mean: {np.mean(valid_before):.3f}")
print(f"  Std:  {np.std(valid_before):.3f}")
print(f"  Min:  {np.min(valid_before):.3f}")
print(f"  Max:  {np.max(valid_before):.3f}")

print(f"\n距离后一个IMU的时间差 (ms):")
print(f"  Mean: {np.mean(valid_after):.3f}")
print(f"  Std:  {np.std(valid_after):.3f}")
print(f"  Min:  {np.min(valid_after):.3f}")
print(f"  Max:  {np.max(valid_after):.3f}")

# 判断是否系统性
print("\n" + "="*70)
print("稳定性判断")
print("="*70)

std_before = np.std(valid_before)
std_after = np.std(valid_after)

if std_before < 1.0 and std_after < 1.0:
    print("✅ 时间差非常稳定 (std < 1ms)")
    print("   这是系统性的固定偏移,不影响deskew")
elif std_before < 3.0 and std_after < 3.0:
    print("✅ 时间差较稳定 (std < 3ms)")
    print("   可以接受,对deskew影响很小")
else:
    print("⚠️  时间差有明显波动 (std > 3ms)")
    print("   可能影响deskew质量")

# 检查IMU覆盖
print(f"\n100ms窗口内IMU数量:")
print(f"  Mean: {np.mean(imu_count_in_window):.1f}")
print(f"  Min:  {np.min(imu_count_in_window)}")
print(f"  Max:  {np.max(imu_count_in_window)}")

# 检查是否有帧没有足够的IMU
low_imu_frames = np.sum(imu_count_in_window < 10)
print(f"  IMU不足(<10)的帧数: {low_imu_frames}/{len(lidar_times)}")

# 第一帧特殊分析
print("\n" + "="*70)
print("第一帧分析")
print("="*70)

first_lidar = lidar_times[0]
first_imu = imu_times[0]
print(f"第一个LiDAR时间: {first_lidar:.6f}")
print(f"第一个IMU时间:   {first_imu:.6f}")
print(f"差异: {(first_imu - first_lidar)*1000:.1f} ms")

if first_imu > first_lidar:
    # 找到第一个有IMU覆盖的LiDAR帧
    for i, lt in enumerate(lidar_times):
        if lt > first_imu:
            print(f"\n第一个有IMU覆盖的LiDAR帧: Frame {i}")
            print(f"  时间: {lt:.6f}")
            print(f"  建议: 跳过前{i}帧")
            break

# LiDAR帧间隔分析
print("\n" + "="*70)
print("LiDAR帧间隔分析")
print("="*70)

lidar_dt = np.diff(lidar_times) * 1000  # ms
print(f"LiDAR帧间隔 (ms):")
print(f"  Mean: {np.mean(lidar_dt):.2f}")
print(f"  Std:  {np.std(lidar_dt):.2f}")
print(f"  Min:  {np.min(lidar_dt):.2f}")
print(f"  Max:  {np.max(lidar_dt):.2f}")
print(f"  频率: {1000/np.mean(lidar_dt):.1f} Hz")

# 保存图表
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 图1: 时间差分布
ax1 = axes[0]
ax1.plot(valid_before, 'b-', alpha=0.7, label='Before IMU')
ax1.plot(valid_after, 'r-', alpha=0.7, label='After IMU')
ax1.set_xlabel('LiDAR Frame')
ax1.set_ylabel('Time to nearest IMU (ms)')
ax1.set_title('LiDAR-IMU Time Difference')
ax1.legend()
ax1.grid(True)

# 图2: 时间差直方图
ax2 = axes[1]
ax2.hist(valid_before, bins=50, alpha=0.7, label=f'Before (std={std_before:.2f}ms)')
ax2.hist(valid_after, bins=50, alpha=0.7, label=f'After (std={std_after:.2f}ms)')
ax2.set_xlabel('Time difference (ms)')
ax2.set_ylabel('Count')
ax2.set_title('Time Difference Distribution')
ax2.legend()
ax2.grid(True)

# 图3: IMU覆盖数量
ax3 = axes[2]
ax3.plot(imu_count_in_window, 'g-')
ax3.axhline(y=20, color='r', linestyle='--', label='Expected (20)')
ax3.set_xlabel('LiDAR Frame')
ax3.set_ylabel('IMU count in ±50ms window')
ax3.set_title('IMU Coverage per LiDAR Frame')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig('/tmp/timestamp_analysis.png', dpi=100)
print(f"\n图表已保存到: /tmp/timestamp_analysis.png")

print("\n" + "="*70)
print("分析完成")
print("="*70)
