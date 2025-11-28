#!/usr/bin/env python3
"""
分析GPS信号质量随时间的变化
"""

import rosbag
import numpy as np

bag_file = '/root/autodl-tmp/info_fixed.bag'

print("="*70)
print("GPS 信号质量时间分析")
print("="*70)

bag = rosbag.Bag(bag_file, 'r')

# 记录GPS协方差随时间变化
gps_data = []
first_time = None

for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
    time = msg.header.stamp.to_sec()
    if first_time is None:
        first_time = time

    cov_x = msg.pose.covariance[0]
    cov_y = msg.pose.covariance[7]
    cov_z = msg.pose.covariance[14]

    pos = [msg.pose.pose.position.x,
           msg.pose.pose.position.y,
           msg.pose.pose.position.z]

    gps_data.append({
        'time': time - first_time,
        'cov_x': cov_x,
        'cov_y': cov_y,
        'cov_z': cov_z,
        'pos': pos
    })

bag.close()

# 分析
times = np.array([d['time'] for d in gps_data])
cov_x = np.array([d['cov_x'] for d in gps_data])
cov_y = np.array([d['cov_y'] for d in gps_data])
positions = np.array([d['pos'] for d in gps_data])

print(f"\n总数据点: {len(gps_data)}")
print(f"时间范围: {times[0]:.1f} - {times[-1]:.1f} 秒")

# 按时间段分析
segments = [
    (0, 60, "0-60s"),
    (60, 120, "60-120s"),
    (120, 180, "120-180s"),
    (180, 300, "180-300s"),
    (300, 600, "300-600s"),
    (600, 900, "600-900s")
]

print("\n协方差随时间变化 (gpsCovThreshold=2.0):")
print("-" * 60)
print(f"{'时间段':<15} {'平均CovX':<12} {'平均CovY':<12} {'可用率':<10}")
print("-" * 60)

for start, end, label in segments:
    mask = (times >= start) & (times < end)
    if np.sum(mask) == 0:
        continue

    seg_cov_x = cov_x[mask]
    seg_cov_y = cov_y[mask]

    usable = np.sum((seg_cov_x < 2.0) & (seg_cov_y < 2.0))
    total = len(seg_cov_x)
    ratio = 100 * usable / total

    print(f"{label:<15} {np.mean(seg_cov_x):<12.3f} {np.mean(seg_cov_y):<12.3f} {ratio:.1f}%")

# 找到协方差最低的时间段
min_cov_idx = np.argmin(cov_x + cov_y)
best_time = times[min_cov_idx]
print(f"\n最低协方差时间: {best_time:.1f}s")
print(f"  CovX: {cov_x[min_cov_idx]:.3f}, CovY: {cov_y[min_cov_idx]:.3f}")

# 检查位置变化
print("\n位置变化分析:")
print(f"  起始位置: [{positions[0,0]:.2f}, {positions[0,1]:.2f}, {positions[0,2]:.2f}]")
print(f"  结束位置: [{positions[-1,0]:.2f}, {positions[-1,1]:.2f}, {positions[-1,2]:.2f}]")
print(f"  总移动距离: {np.linalg.norm(positions[-1] - positions[0]):.2f} m")

# 检查大的位置跳变
pos_diff = np.diff(positions, axis=0)
distances = np.linalg.norm(pos_diff, axis=1)
large_jumps = np.where(distances > 1.0)[0]

print(f"\n位置跳变 (>1m): {len(large_jumps)} 次")
if len(large_jumps) > 0 and len(large_jumps) <= 10:
    for idx in large_jumps[:5]:
        print(f"  t={times[idx]:.1f}s: {distances[idx]:.2f}m")

print("\n" + "="*70)
