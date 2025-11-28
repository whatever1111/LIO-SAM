#!/usr/bin/env python3
"""
分析GPS恢复时刻的位置差异
"""

import rosbag
import numpy as np

bag_file = '/root/autodl-tmp/info_fixed.bag'

print("="*70)
print("GPS恢复时刻分析")
print("="*70)

bag = rosbag.Bag(bag_file, 'r')

# 收集前100秒的GPS数据
gps_data = []
first_time = None

for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
    time = msg.header.stamp.to_sec()
    if first_time is None:
        first_time = time

    rel_time = time - first_time
    if rel_time > 100:
        break

    cov_x = msg.pose.covariance[0]
    cov_y = msg.pose.covariance[7]

    pos = np.array([msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z])

    gps_data.append({
        'time': rel_time,
        'abs_time': time,
        'cov': (cov_x + cov_y) / 2,
        'pos': pos
    })

bag.close()

# 找到GPS恢复点 (协方差从高变低)
print("\nGPS恢复时刻检测:")
threshold = 2.0
recovery_idx = None

for i in range(1, len(gps_data)):
    prev_cov = gps_data[i-1]['cov']
    curr_cov = gps_data[i]['cov']

    if prev_cov > threshold and curr_cov < threshold:
        recovery_idx = i
        print(f"  检测到GPS恢复!")
        print(f"  时间: {gps_data[i]['time']:.1f}s")
        print(f"  协方差变化: {prev_cov:.3f} → {curr_cov:.3f}")
        break

if recovery_idx is None:
    print("  未检测到明显的GPS恢复点")
    # 找第一个低协方差点
    for i, d in enumerate(gps_data):
        if d['cov'] < threshold:
            recovery_idx = i
            print(f"  第一个可用GPS: {d['time']:.1f}s, cov={d['cov']:.3f}")
            break

# 分析恢复前后的位置
if recovery_idx:
    # 恢复前的位置 (最后一个高协方差)
    before_data = gps_data[recovery_idx - 1] if recovery_idx > 0 else None
    # 恢复后的位置 (第一个低协方差)
    after_data = gps_data[recovery_idx]

    print("\n位置分析:")
    if before_data:
        print(f"  恢复前 (t={before_data['time']:.1f}s):")
        print(f"    GPS位置: [{before_data['pos'][0]:.2f}, {before_data['pos'][1]:.2f}, {before_data['pos'][2]:.2f}]")
        print(f"    协方差: {before_data['cov']:.3f}")

    print(f"\n  恢复后 (t={after_data['time']:.1f}s):")
    print(f"    GPS位置: [{after_data['pos'][0]:.2f}, {after_data['pos'][1]:.2f}, {after_data['pos'][2]:.2f}]")
    print(f"    协方差: {after_data['cov']:.3f}")

    # 计算前60秒内GPS位置的实际移动
    early_gps = [d for d in gps_data if d['time'] < 60]
    if len(early_gps) > 10:
        start_pos = early_gps[0]['pos']
        end_pos = early_gps[-1]['pos']
        gps_movement = np.linalg.norm(end_pos - start_pos)
        print(f"\n  前60秒GPS记录的移动: {gps_movement:.2f}m")
        print(f"    起点: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
        print(f"    终点: [{end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}]")

# 关键点:检查addGPSFactor的条件
print("\n" + "="*70)
print("LIO-SAM GPS因子添加条件检查")
print("="*70)

print("""
addGPSFactor() 条件:
1. cloudKeyPoses3D 不为空
2. pointDistance(front, back) >= 1.0m  (修改后)
3. poseCovariance(3,3) >= poseCovThreshold 或 poseCovariance(4,4) >= poseCovThreshold
4. gpsCovThreshold >= gpsMsg协方差

当GPS恢复时:
- 条件4: 协方差从 4.2 降到 0.15, 开始满足
- 此时LiDAR里程计已经累积了漂移
- GPS因子强制拉回 → 位置跳变
""")

print("="*70)
