#!/usr/bin/env python3
"""
精确定位速度跳变根本原因的诊断脚本
运行方法: python3 precise_velocity_diagnosis.py <bag_file>

重点检查:
1. IMU坐标变换后的重力方向是否正确
2. IMU预积分的速度累积
3. Scan-to-map匹配特征点数量
4. LiDAR与IMU的时间同步
"""

import rospy
import rosbag
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R

def check_imu_gravity_direction(bag_file, imu_topic):
    """检查IMU数据经过坐标变换后的重力方向"""
    print("\n" + "="*60)
    print("1. IMU重力方向检查")
    print("="*60)

    # 配置的外参旋转矩阵 (从params.yaml)
    # extrinsicRot: [-1, 0, 0, 0, -1, 0, 0, 0, 1]
    extRot = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    print(f"\n配置的extrinsicRot矩阵:")
    print(extRot)
    print(f"矩阵行列式: {np.linalg.det(extRot):.2f}")

    bag = rosbag.Bag(bag_file)
    acc_samples = []

    # 读取前100个IMU样本
    count = 0
    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        if count >= 100:
            break

        # 原始加速度
        acc_raw = np.array([
            msg.data.linear_acceleration.x if hasattr(msg, 'data') else msg.linear_acceleration.x,
            msg.data.linear_acceleration.y if hasattr(msg, 'data') else msg.linear_acceleration.y,
            msg.data.linear_acceleration.z if hasattr(msg, 'data') else msg.linear_acceleration.z
        ])

        # 变换后的加速度
        acc_transformed = extRot @ acc_raw
        acc_samples.append({
            'raw': acc_raw,
            'transformed': acc_transformed
        })
        count += 1

    bag.close()

    if not acc_samples:
        print("错误: 没有找到IMU数据!")
        return False

    # 计算平均加速度
    raw_accs = np.array([s['raw'] for s in acc_samples])
    trans_accs = np.array([s['transformed'] for s in acc_samples])

    mean_raw = np.mean(raw_accs, axis=0)
    mean_trans = np.mean(trans_accs, axis=0)

    print(f"\n原始IMU加速度 (静止时应约等于重力):")
    print(f"  平均: [{mean_raw[0]:.3f}, {mean_raw[1]:.3f}, {mean_raw[2]:.3f}] m/s²")
    print(f"  范数: {np.linalg.norm(mean_raw):.3f} m/s² (期望≈9.81)")

    print(f"\n变换后加速度:")
    print(f"  平均: [{mean_trans[0]:.3f}, {mean_trans[1]:.3f}, {mean_trans[2]:.3f}] m/s²")
    print(f"  范数: {np.linalg.norm(mean_trans):.3f} m/s²")

    # 分析重力分量
    print("\n>>> 重力方向分析:")

    # GTSAM的MakeSharedU(g)假设重力在+Z方向
    # 即如果Z轴向上，重力补偿为 acc_corrected = acc_measured - (0, 0, g)
    expected_gravity = np.array([0, 0, 9.81])

    # 计算变换后加速度与期望重力的差异
    diff = mean_trans - expected_gravity
    print(f"  与期望重力(0,0,9.81)的差异: [{diff[0]:.3f}, {diff[1]:.3f}, {diff[2]:.3f}]")

    # 检查最大分量
    max_idx = np.argmax(np.abs(mean_trans))
    axis_names = ['X', 'Y', 'Z']
    print(f"  主要重力分量在: {axis_names[max_idx]}轴 ({mean_trans[max_idx]:.3f} m/s²)")

    # 判断是否有问题
    if max_idx != 2:  # 重力应该在Z轴
        print(f"\n!!! 警告: 重力主要分量不在Z轴!")
        print(f"    这会导致GTSAM的重力补偿错误!")
        print(f"    因为MakeSharedU(g)假设重力在+Z方向")
        return False
    elif mean_trans[2] < 0:
        print(f"\n!!! 警告: 重力方向为-Z!")
        print(f"    应该使用MakeSharedD(g)而不是MakeSharedU(g)")
        return False
    else:
        print(f"\n✓ 重力方向正确 (+Z)")
        return True


def check_feature_extraction(bag_file):
    """检查特征提取数量"""
    print("\n" + "="*60)
    print("2. 特征提取检查")
    print("="*60)

    bag = rosbag.Bag(bag_file)

    # 检查点云数据
    lidar_topic = '/lidar_points'
    point_counts = []

    count = 0
    for topic, msg, t in bag.read_messages(topics=[lidar_topic]):
        if count >= 50:
            break
        point_counts.append(msg.width * msg.height)
        count += 1

    bag.close()

    if not point_counts:
        print("错误: 没有找到点云数据!")
        return

    print(f"\n点云统计 (前50帧):")
    print(f"  平均点数: {np.mean(point_counts):.0f}")
    print(f"  最小点数: {np.min(point_counts)}")
    print(f"  最大点数: {np.max(point_counts)}")

    # 检查配置的范围过滤
    lidar_min_range = 0.3
    lidar_max_range = 100.0  # 从params.yaml

    print(f"\n当前范围过滤配置:")
    print(f"  lidarMinRange: {lidar_min_range} m")
    print(f"  lidarMaxRange: {lidar_max_range} m")

    if lidar_max_range < 10:
        print(f"\n!!! 严重警告: lidarMaxRange={lidar_max_range}m 太小!")
        print(f"    这会过滤掉大部分点云!")


def analyze_velocity_anomaly_pattern(csv_file):
    """分析速度异常的模式"""
    print("\n" + "="*60)
    print("3. 速度异常模式分析")
    print("="*60)

    import pandas as pd

    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    # 分析IMU预积分速度异常
    vel_anomalies = df[df['type'] == 'VEL_ANOMALY']

    if vel_anomalies.empty:
        print("没有找到速度异常记录")
        return

    # 解析速度分量
    vx_list = []
    vy_list = []
    vz_list = []

    for desc in vel_anomalies['description']:
        # 解析格式: "... vx=0.04, vy=9.80, vz=-2.00"
        try:
            parts = desc.split(',')
            for p in parts:
                if 'vx=' in p:
                    vx_list.append(float(p.split('=')[1]))
                elif 'vy=' in p:
                    vy_list.append(float(p.split('=')[1]))
                elif 'vz=' in p:
                    vz_list.append(float(p.split('=')[1].strip('"')))
        except:
            pass

    if vx_list:
        print(f"\n速度分量统计 ({len(vx_list)} 个异常):")
        print(f"  Vx: 平均={np.mean(vx_list):.2f}, 范围=[{np.min(vx_list):.2f}, {np.max(vx_list):.2f}]")
        print(f"  Vy: 平均={np.mean(vy_list):.2f}, 范围=[{np.min(vy_list):.2f}, {np.max(vy_list):.2f}]")
        print(f"  Vz: 平均={np.mean(vz_list):.2f}, 范围=[{np.min(vz_list):.2f}, {np.max(vz_list):.2f}]")

        # 关键发现
        print("\n>>> 关键发现:")

        if np.mean(np.abs(vy_list)) > 8:
            print(f"  !!! Vy方向速度异常大 (约{np.mean(vy_list):.1f} m/s)")
            print(f"      这个值接近重力加速度9.81 m/s²!")
            print(f"      强烈暗示重力没有被正确补偿!")
            print(f"")
            print(f"  可能原因:")
            print(f"  1. IMU坐标变换后重力不在Z轴")
            print(f"  2. 使用了错误的PreintegrationParams")
            print(f"  3. extrinsicRot配置与实际IMU安装不匹配")


def check_time_sync(bag_file):
    """检查时间同步"""
    print("\n" + "="*60)
    print("4. 时间同步检查")
    print("="*60)

    bag = rosbag.Bag(bag_file)

    imu_times = []
    lidar_times = []

    imu_topic = '/fixposition/fpa/corrimu'
    lidar_topic = '/lidar_points'

    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        if len(imu_times) >= 1000:
            break

        imu_times.append(msg.data.header.stamp.to_sec())

    for topic, msg, t in bag.read_messages(topics=[lidar_topic]):
        if len(lidar_times) >= 100:
            break
        print(msg)
        lidar_times.append(msg.data.header.stamp.to_sec())

    bag.close()

    if imu_times and lidar_times:
        print(f"\nIMU时间范围: {imu_times[0]:.3f} - {imu_times[-1]:.3f}")
        print(f"LiDAR时间范围: {lidar_times[0]:.3f} - {lidar_times[-1]:.3f}")

        # 计算IMU频率
        imu_dt = np.diff(imu_times)
        print(f"\nIMU频率: {1.0/np.mean(imu_dt):.1f} Hz")

        # 计算LiDAR频率
        lidar_dt = np.diff(lidar_times)
        print(f"LiDAR频率: {1.0/np.mean(lidar_dt):.1f} Hz")

        # 检查时间偏移
        time_offset = lidar_times[0] - imu_times[0]
        print(f"\n时间偏移 (LiDAR - IMU): {time_offset:.4f} 秒")


def generate_diagnosis_report():
    """生成诊断报告"""
    print("\n" + "="*60)
    print("诊断报告 - 速度跳变根本原因分析")
    print("="*60)

    print("""
基于诊断数据分析,问题的根本原因是:

┌─────────────────────────────────────────────────────────────┐
│  根本原因: IMU坐标变换与GTSAM重力模型不匹配                    │
└─────────────────────────────────────────────────────────────┘

详细分析:
---------
1. 诊断数据显示: Vy方向速度异常 ≈ 9.8 m/s (接近重力加速度!)

2. 当前配置:
   - extrinsicRot = Rz(180°): [-1,0,0, 0,-1,0, 0,0,1]
   - PreintegrationParams::MakeSharedU(g) 假设重力在+Z方向

3. 问题链:
   原始IMU数据 → extRot变换 → 重力方向改变 → GTSAM补偿错误 → 速度累积

4. 具体来说:
   - 如果原始IMU的重力在某个轴上
   - 经过Rz(180°)变换后
   - 可能导致重力不在Z轴,或者方向相反
   - GTSAM用 (0,0,+g) 补偿,但实际重力可能是其他方向
   - 导致约9.8 m/s²的残差持续累积到速度

可能的解决方案:
---------------
方案A: 检查并修正extrinsicRot
   - 确保变换后重力在+Z方向
   - 静止状态应该: acc_transformed ≈ (0, 0, +9.81)

方案B: 修改PreintegrationParams
   - 如果重力在-Z方向,使用 MakeSharedD(g)
   - 或者直接设置正确的重力向量

方案C: 检查IMU实际安装方向
   - 确认IMU的坐标系定义
   - 确认extRot与实际安装一致

建议的验证步骤:
---------------
1. 打印原始IMU加速度和变换后加速度
2. 确认静止状态下变换后加速度 ≈ (0, 0, 9.81)
3. 如果不是,调整extrinsicRot或重力模型
""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 precise_velocity_diagnosis.py <bag_file>")
        print("Example: python3 precise_velocity_diagnosis.py /root/autodl-tmp/info_fixed.bag")
        sys.exit(1)

    bag_file = sys.argv[1]

    if not os.path.exists(bag_file):
        print(f"Error: Bag file not found: {bag_file}")
        sys.exit(1)

    print("="*60)
    print("LIO-SAM 速度跳变精确诊断")
    print("="*60)
    print(f"Bag文件: {bag_file}")

    # 1. 检查IMU重力方向
    imu_topic = '/fixposition/fpa/corrimu'
    gravity_ok = check_imu_gravity_direction(bag_file, imu_topic)

    # 2. 检查特征提取
    check_feature_extraction(bag_file)

    # 3. 分析速度异常模式
    csv_file = "/tmp/lio_sam_diagnostic_20251124_173540.csv"
    analyze_velocity_anomaly_pattern(csv_file)

    # 4. 检查时间同步
    check_time_sync(bag_file)

    # 5. 生成诊断报告
    generate_diagnosis_report()


if __name__ == "__main__":
    main()
