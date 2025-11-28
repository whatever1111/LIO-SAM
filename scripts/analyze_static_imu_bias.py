#!/usr/bin/env python3
"""
分析bag文件静止阶段的IMU偏差
"""

import rosbag
import numpy as np
import sys

def analyze_static_phase(bag_file, static_duration=30):
    """分析静止阶段的IMU数据"""

    print("="*70)
    print("静止阶段IMU偏差分析")
    print("="*70)

    extRot = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    bag = rosbag.Bag(bag_file)
    imu_topic = '/fixposition/fpa/corrimu'

    accs = []
    gyros = []
    timestamps = []

    first_time = None
    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        imu_time = msg.data.header.stamp.to_sec()

        if first_time is None:
            first_time = imu_time

        # 只取前N秒
        if imu_time - first_time > static_duration:
            break

        raw_acc = np.array([
            msg.data.linear_acceleration.x,
            msg.data.linear_acceleration.y,
            msg.data.linear_acceleration.z
        ])

        raw_gyro = np.array([
            msg.data.angular_velocity.x,
            msg.data.angular_velocity.y,
            msg.data.angular_velocity.z
        ])

        transformed_acc = extRot @ raw_acc
        transformed_gyro = extRot @ raw_gyro

        accs.append(transformed_acc)
        gyros.append(transformed_gyro)
        timestamps.append(imu_time)

    bag.close()

    accs = np.array(accs)
    gyros = np.array(gyros)

    print(f"\n分析前 {static_duration} 秒的数据 ({len(accs)} 个样本)")

    # 加速度分析
    print("\n" + "-"*70)
    print("加速度偏差 (变换后)")
    print("-"*70)

    acc_mean = np.mean(accs, axis=0)
    acc_std = np.std(accs, axis=0)

    print(f"平均值: [{acc_mean[0]:.4f}, {acc_mean[1]:.4f}, {acc_mean[2]:.4f}] m/s²")
    print(f"标准差: [{acc_std[0]:.4f}, {acc_std[1]:.4f}, {acc_std[2]:.4f}] m/s²")

    # 与理想重力的偏差
    ideal_gravity = np.array([0, 0, 9.81])
    bias = acc_mean - ideal_gravity

    print(f"\n与理想重力(0,0,9.81)的偏差:")
    print(f"  X偏差: {bias[0]:.4f} m/s²")
    print(f"  Y偏差: {bias[1]:.4f} m/s²")
    print(f"  Z偏差: {bias[2]:.4f} m/s²")

    # 陀螺仪分析
    print("\n" + "-"*70)
    print("陀螺仪偏差 (变换后)")
    print("-"*70)

    gyro_mean = np.mean(gyros, axis=0)
    gyro_std = np.std(gyros, axis=0)

    print(f"平均值: [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}] rad/s")
    print(f"标准差: [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}] rad/s")

    # 计算累积影响
    print("\n" + "-"*70)
    print("误差累积预测")
    print("-"*70)

    print(f"\n如果不补偿加速度偏差:")
    for t in [1, 5, 10, 30, 60]:
        vel_x = bias[0] * t
        vel_y = bias[1] * t
        vel_z = bias[2] * t
        print(f"  {t:2d}秒后速度: [{vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f}] m/s")

    # 建议
    print("\n" + "="*70)
    print("建议的修复")
    print("="*70)

    print(f"\n在imuPreintegration.cpp中修改初始偏差:")
    print(f"```cpp")
    print(f"// 原来: 全零偏差")
    print(f"// gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());")
    print(f"")
    print(f"// 修改为: 预设加速度偏差 (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)")
    print(f"gtsam::imuBias::ConstantBias prior_imu_bias(")
    print(f"    (gtsam::Vector(6) << {bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}, ")
    print(f"                         {gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}).finished());")
    print(f"```")

    return {
        'acc_bias': bias,
        'gyro_bias': gyro_mean
    }

if __name__ == "__main__":
    bag_file = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/info_fixed.bag"
    results = analyze_static_phase(bag_file, static_duration=30)
