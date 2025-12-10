#!/usr/bin/env python3
"""
航向角深度分析脚本
分析GPS航向与LIO-SAM融合航向的差异
"""

import rosbag
import numpy as np
import math

BAG_PATH = '/root/autodl-tmp/data/hkust_20201105_full.bag'
GPS_TOPIC = '/odometry/gps'
FUSION_TOPIC = '/lio_sam/mapping/odometry'

def normalize_angle(angle):
    """将角度归一化到 [-180, 180]"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def compute_heading_from_velocity(vx, vy):
    """从速度计算航向角（度），返回 [-180, 180]"""
    if abs(vx) < 0.01 and abs(vy) < 0.01:
        return None  # 速度太小，航向不可靠
    return math.degrees(math.atan2(vy, vx))

def compute_heading_from_quaternion(qx, qy, qz, qw):
    """从四元数提取yaw角（度）"""
    # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw)

def main():
    print("=" * 70)
    print("航向角深度分析")
    print("=" * 70)

    # 读取数据
    gps_data = []
    fusion_data = []

    bag = rosbag.Bag(BAG_PATH, 'r')

    for topic, msg, t in bag.read_messages(topics=[GPS_TOPIC, FUSION_TOPIC]):
        ts = msg.header.stamp.to_sec()
        if topic == GPS_TOPIC:
            gps_data.append({
                'time': ts,
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'qx': msg.pose.pose.orientation.x,
                'qy': msg.pose.pose.orientation.y,
                'qz': msg.pose.pose.orientation.z,
                'qw': msg.pose.pose.orientation.w,
            })
        elif topic == FUSION_TOPIC:
            fusion_data.append({
                'time': ts,
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'qx': msg.pose.pose.orientation.x,
                'qy': msg.pose.pose.orientation.y,
                'qz': msg.pose.pose.orientation.z,
                'qw': msg.pose.pose.orientation.w,
            })

    bag.close()

    print(f"\nGPS数据点: {len(gps_data)}")
    print(f"Fusion数据点: {len(fusion_data)}")

    if len(gps_data) < 2 or len(fusion_data) < 2:
        print("数据不足")
        return

    # 时间范围
    gps_t0 = gps_data[0]['time']
    fusion_t0 = fusion_data[0]['time']
    t0 = max(gps_t0, fusion_t0)

    print(f"\n时间基准: {t0:.2f}")

    # 1. 分析GPS四元数航向
    print("\n" + "=" * 70)
    print("1. GPS四元数航向分析")
    print("=" * 70)

    gps_quat_headings = []
    for d in gps_data[:50]:  # 前50个点
        heading = compute_heading_from_quaternion(d['qx'], d['qy'], d['qz'], d['qw'])
        gps_quat_headings.append(heading)

    if gps_quat_headings:
        print(f"GPS四元数航向（前10个）:")
        for i, h in enumerate(gps_quat_headings[:10]):
            print(f"  [{i}] {h:.1f}°")
        print(f"GPS四元数航向平均: {np.mean(gps_quat_headings):.1f}°")
        print(f"GPS四元数航向标准差: {np.std(gps_quat_headings):.1f}°")

    # 2. 分析Fusion四元数航向
    print("\n" + "=" * 70)
    print("2. Fusion四元数航向分析")
    print("=" * 70)

    fusion_quat_headings = []
    for d in fusion_data[:50]:
        heading = compute_heading_from_quaternion(d['qx'], d['qy'], d['qz'], d['qw'])
        fusion_quat_headings.append(heading)

    if fusion_quat_headings:
        print(f"Fusion四元数航向（前10个）:")
        for i, h in enumerate(fusion_quat_headings[:10]):
            print(f"  [{i}] {h:.1f}°")
        print(f"Fusion四元数航向平均: {np.mean(fusion_quat_headings):.1f}°")
        print(f"Fusion四元数航向标准差: {np.std(fusion_quat_headings):.1f}°")

    # 3. 计算速度航向
    print("\n" + "=" * 70)
    print("3. GPS速度航向分析")
    print("=" * 70)

    gps_vel_headings = []
    for i in range(1, min(51, len(gps_data))):
        dt = gps_data[i]['time'] - gps_data[i-1]['time']
        if dt > 0.001:
            vx = (gps_data[i]['x'] - gps_data[i-1]['x']) / dt
            vy = (gps_data[i]['y'] - gps_data[i-1]['y']) / dt
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > 0.5:  # 速度大于0.5m/s
                heading = compute_heading_from_velocity(vx, vy)
                if heading is not None:
                    gps_vel_headings.append(heading)

    if gps_vel_headings:
        print(f"GPS速度航向（有效点数: {len(gps_vel_headings)}）:")
        for i, h in enumerate(gps_vel_headings[:10]):
            print(f"  [{i}] {h:.1f}°")
        print(f"GPS速度航向平均: {np.mean(gps_vel_headings):.1f}°")

    # 4. Fusion速度航向
    print("\n" + "=" * 70)
    print("4. Fusion速度航向分析")
    print("=" * 70)

    fusion_vel_headings = []
    for i in range(1, min(51, len(fusion_data))):
        dt = fusion_data[i]['time'] - fusion_data[i-1]['time']
        if dt > 0.001:
            vx = (fusion_data[i]['x'] - fusion_data[i-1]['x']) / dt
            vy = (fusion_data[i]['y'] - fusion_data[i-1]['y']) / dt
            speed = math.sqrt(vx*vx + vy*vy)
            if speed > 0.5:
                heading = compute_heading_from_velocity(vx, vy)
                if heading is not None:
                    fusion_vel_headings.append(heading)

    if fusion_vel_headings:
        print(f"Fusion速度航向（有效点数: {len(fusion_vel_headings)}）:")
        for i, h in enumerate(fusion_vel_headings[:10]):
            print(f"  [{i}] {h:.1f}°")
        print(f"Fusion速度航向平均: {np.mean(fusion_vel_headings):.1f}°")

    # 5. 航向差异分析
    print("\n" + "=" * 70)
    print("5. 航向差异分析")
    print("=" * 70)

    if gps_quat_headings and fusion_quat_headings:
        quat_diff = normalize_angle(np.mean(fusion_quat_headings) - np.mean(gps_quat_headings))
        print(f"四元数航向差异（Fusion - GPS）: {quat_diff:.1f}°")

    if gps_vel_headings and fusion_vel_headings:
        vel_diff = normalize_angle(np.mean(fusion_vel_headings) - np.mean(gps_vel_headings))
        print(f"速度航向差异（Fusion - GPS）: {vel_diff:.1f}°")

    if gps_quat_headings and gps_vel_headings:
        gps_qv_diff = normalize_angle(np.mean(gps_quat_headings) - np.mean(gps_vel_headings))
        print(f"GPS内部差异（四元数 - 速度）: {gps_qv_diff:.1f}°")

    if fusion_quat_headings and fusion_vel_headings:
        fusion_qv_diff = normalize_angle(np.mean(fusion_quat_headings) - np.mean(fusion_vel_headings))
        print(f"Fusion内部差异（四元数 - 速度）: {fusion_qv_diff:.1f}°")

    # 6. 时间序列航向对比（每10秒采样）
    print("\n" + "=" * 70)
    print("6. 时间序列航向对比")
    print("=" * 70)

    # 按时间窗口分析
    time_windows = [0, 30, 60, 90, 120, 150, 180]

    print(f"{'时间(s)':<10} {'GPS四元数':>12} {'GPS速度':>12} {'Fusion四元数':>14} {'Fusion速度':>12}")
    print("-" * 70)

    for tw in time_windows:
        # GPS数据
        gps_qh = None
        gps_vh = None
        for i, d in enumerate(gps_data):
            if d['time'] - gps_t0 >= tw and d['time'] - gps_t0 < tw + 5:
                gps_qh = compute_heading_from_quaternion(d['qx'], d['qy'], d['qz'], d['qw'])
                if i > 0:
                    dt = d['time'] - gps_data[i-1]['time']
                    if dt > 0.001:
                        vx = (d['x'] - gps_data[i-1]['x']) / dt
                        vy = (d['y'] - gps_data[i-1]['y']) / dt
                        if math.sqrt(vx*vx + vy*vy) > 0.3:
                            gps_vh = compute_heading_from_velocity(vx, vy)
                break

        # Fusion数据
        fus_qh = None
        fus_vh = None
        for i, d in enumerate(fusion_data):
            if d['time'] - fusion_t0 >= tw and d['time'] - fusion_t0 < tw + 5:
                fus_qh = compute_heading_from_quaternion(d['qx'], d['qy'], d['qz'], d['qw'])
                if i > 0:
                    dt = d['time'] - fusion_data[i-1]['time']
                    if dt > 0.001:
                        vx = (d['x'] - fusion_data[i-1]['x']) / dt
                        vy = (d['y'] - fusion_data[i-1]['y']) / dt
                        if math.sqrt(vx*vx + vy*vy) > 0.3:
                            fus_vh = compute_heading_from_velocity(vx, vy)
                break

        gps_qh_str = f"{gps_qh:.1f}°" if gps_qh is not None else "N/A"
        gps_vh_str = f"{gps_vh:.1f}°" if gps_vh is not None else "N/A"
        fus_qh_str = f"{fus_qh:.1f}°" if fus_qh is not None else "N/A"
        fus_vh_str = f"{fus_vh:.1f}°" if fus_vh is not None else "N/A"

        print(f"{tw:<10} {gps_qh_str:>12} {gps_vh_str:>12} {fus_qh_str:>14} {fus_vh_str:>12}")

    # 7. 坐标系分析
    print("\n" + "=" * 70)
    print("7. 坐标系分析")
    print("=" * 70)

    print("""
当前配置:
- gpsExtrinsicRot: [0,1,0; -1,0,0; 0,0,1] (90°逆时针旋转)
  含义: X_lidar = Y_gps, Y_lidar = -X_gps

- LIO-SAM初始航向: Yaw=0 (假设朝北)
- GPS ENU坐标系: +X=East, +Y=North

如果Fusion航向 - GPS航向 ≈ 90°:
  说明GPS坐标没有正确旋转到LIO-SAM坐标系

如果Fusion航向 - GPS航向 ≈ 0°:
  说明坐标系对齐正确

如果差异不稳定（随时间变化大）:
  说明LIO-SAM本身的航向估计有漂移问题
""")

    # 8. 建议
    print("\n" + "=" * 70)
    print("8. 诊断建议")
    print("=" * 70)

    if fusion_quat_headings and gps_quat_headings:
        diff = abs(normalize_angle(np.mean(fusion_quat_headings) - np.mean(gps_quat_headings)))

        if diff < 15:
            print("四元数航向差异 < 15°: 坐标系基本对齐")
        elif diff > 75 and diff < 105:
            print(f"四元数航向差异 ≈ {diff:.0f}°: 存在约90°偏差")
            print("建议: 检查gpsExtrinsicRot配置是否正确")
            print("      或者检查IMU初始航向是否正确")
        elif diff > 165 or diff < -165:
            print("四元数航向差异 ≈ 180°: X/Y轴方向相反")
        else:
            print(f"四元数航向差异 = {diff:.0f}°: 非典型偏差，需要进一步分析")

if __name__ == '__main__':
    main()
