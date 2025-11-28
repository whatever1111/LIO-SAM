#!/usr/bin/env python3
"""
系统性分析rosbag中的IMU数据，确认Y方向加速度偏差
"""

import rosbag
import numpy as np
import sys
import os

def analyze_imu_bias(bag_file):
    """分析整个bag文件的IMU数据"""

    print("="*70)
    print("IMU加速度偏差系统性分析")
    print("="*70)
    print(f"Bag文件: {bag_file}")

    # 外参旋转矩阵 (从params.yaml)
    extRot = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    bag = rosbag.Bag(bag_file)

    # 收集所有IMU数据
    imu_topic = '/fixposition/fpa/corrimu'

    raw_accs = []
    transformed_accs = []
    timestamps = []

    print(f"\n读取IMU数据 (topic: {imu_topic})...")

    count = 0
    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        raw_acc = np.array([
            msg.data.linear_acceleration.x,
            msg.data.linear_acceleration.y,
            msg.data.linear_acceleration.z
        ])

        transformed_acc = extRot @ raw_acc

        raw_accs.append(raw_acc)
        transformed_accs.append(transformed_acc)
        timestamps.append(msg.data.header.stamp.to_sec())

        count += 1
        if count % 10000 == 0:
            print(f"  已处理 {count} 条消息...")

    bag.close()

    raw_accs = np.array(raw_accs)
    transformed_accs = np.array(transformed_accs)
    timestamps = np.array(timestamps)

    print(f"\n总共 {len(raw_accs)} 条IMU消息")
    print(f"时间范围: {timestamps[0]:.2f} - {timestamps[-1]:.2f} ({timestamps[-1]-timestamps[0]:.1f}秒)")

    # 整体统计
    print("\n" + "-"*70)
    print("1. 整体统计 (变换后加速度)")
    print("-"*70)

    mean_acc = np.mean(transformed_accs, axis=0)
    std_acc = np.std(transformed_accs, axis=0)

    print(f"平均值: [{mean_acc[0]:.4f}, {mean_acc[1]:.4f}, {mean_acc[2]:.4f}] m/s²")
    print(f"标准差: [{std_acc[0]:.4f}, {std_acc[1]:.4f}, {std_acc[2]:.4f}] m/s²")
    print(f"范数:   {np.linalg.norm(mean_acc):.4f} m/s² (期望≈9.81)")

    # 分时段统计
    print("\n" + "-"*70)
    print("2. 分时段统计 (每30秒)")
    print("-"*70)

    duration = timestamps[-1] - timestamps[0]
    segment_duration = 30.0  # 秒
    num_segments = int(np.ceil(duration / segment_duration))

    segment_means = []

    print(f"{'时段':^15} | {'X平均':^10} | {'Y平均':^10} | {'Z平均':^10} | {'Y偏差':^10}")
    print("-"*70)

    for i in range(num_segments):
        t_start = timestamps[0] + i * segment_duration
        t_end = t_start + segment_duration

        mask = (timestamps >= t_start) & (timestamps < t_end)
        if np.sum(mask) < 10:
            continue

        segment_acc = transformed_accs[mask]
        segment_mean = np.mean(segment_acc, axis=0)
        segment_means.append(segment_mean)

        print(f"{i*30:3d}-{(i+1)*30:3d}s       | {segment_mean[0]:10.4f} | {segment_mean[1]:10.4f} | {segment_mean[2]:10.4f} | {segment_mean[1]:10.4f}")

    segment_means = np.array(segment_means)

    # Y方向偏差分析
    print("\n" + "-"*70)
    print("3. Y方向偏差详细分析")
    print("-"*70)

    y_mean = mean_acc[1]
    y_std = std_acc[1]
    y_min = np.min(transformed_accs[:, 1])
    y_max = np.max(transformed_accs[:, 1])

    print(f"Y方向加速度:")
    print(f"  平均值: {y_mean:.4f} m/s²")
    print(f"  标准差: {y_std:.4f} m/s²")
    print(f"  范围:   [{y_min:.4f}, {y_max:.4f}] m/s²")

    # 计算倾斜角
    tilt_angle = np.arctan2(y_mean, mean_acc[2]) * 180 / np.pi
    print(f"\n推算的倾斜角: {tilt_angle:.2f}° (Y-Z平面)")

    # 分析Y偏差的稳定性
    if len(segment_means) > 1:
        y_segment_std = np.std(segment_means[:, 1])
        print(f"Y偏差的时段间标准差: {y_segment_std:.4f} m/s²")

        if y_segment_std < 0.05:
            print("  → Y偏差非常稳定 (适合预设为常数偏差)")
        else:
            print("  → Y偏差有变化 (可能需要动态估计)")

    # 结论
    print("\n" + "="*70)
    print("4. 结论")
    print("="*70)

    if abs(y_mean) > 0.1:
        print(f"\n!!! 确认: Y方向存在 {y_mean:.4f} m/s² 的系统性加速度偏差 !!!")
        print(f"\n这个偏差会导致:")
        print(f"  - 每秒累积 {y_mean:.2f} m/s 的Y方向速度")
        print(f"  - 10秒后累积 {y_mean*10:.1f} m/s 的速度")
        print(f"  - 100秒后累积 {y_mean*100:.1f} m/s 的速度")

        print(f"\n建议的修复方案:")
        print(f"  在imuPreintegration.cpp中预设加速度偏差:")
        print(f"  gtsam::imuBias::ConstantBias prior_imu_bias(")
        print(f"      (gtsam::Vector(6) << 0, {y_mean:.4f}, 0, 0, 0, 0).finished());")
    else:
        print(f"\nY方向偏差较小 ({y_mean:.4f} m/s²)，可能不是主要问题")

    # 检查X和Z方向
    print("\n" + "-"*70)
    print("5. X和Z方向检查")
    print("-"*70)

    x_mean = mean_acc[0]
    z_mean = mean_acc[2]

    print(f"X方向偏差: {x_mean:.4f} m/s²", end="")
    if abs(x_mean) > 0.1:
        print(" (可能也需要补偿)")
    else:
        print(" (可忽略)")

    print(f"Z方向平均: {z_mean:.4f} m/s² (期望≈9.81)", end="")
    if abs(z_mean - 9.81) > 0.1:
        print(" (可能需要校准imuGravity)")
    else:
        print(" (正常)")

    return {
        'y_bias': y_mean,
        'x_bias': x_mean,
        'z_mean': z_mean,
        'tilt_angle': tilt_angle
    }

def main():
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]
    else:
        bag_file = "/root/autodl-tmp/info_fixed.bag"

    if not os.path.exists(bag_file):
        print(f"错误: 文件不存在: {bag_file}")
        sys.exit(1)

    results = analyze_imu_bias(bag_file)

    print("\n" + "="*70)
    print("分析完成")
    print("="*70)

if __name__ == "__main__":
    main()
