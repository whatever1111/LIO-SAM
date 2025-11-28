#!/usr/bin/env python3
"""
基于实际使用的ECEF数据测试坐标转换
只使用 /fixposition/fpa/odometry (ECEF) 数据
"""

import numpy as np
import rosbag

def ecef_to_lla(ecef):
    x, y, z = ecef
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f*f
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return lat, lon

def get_ecef_to_enu_matrix(lat, lon):
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    return np.array([
        [-sin_lon,          cos_lon,          0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
        [cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])

def get_yaw_rotation_matrix(yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def test_ecef_data(bag_file):
    print("=" * 70)
    print("基于ECEF数据的坐标转换测试")
    print("=" * 70)

    bag = rosbag.Bag(bag_file, 'r')

    # 只读取实际使用的ECEF数据
    ecef_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        ecef_data.append({
            'time': msg.header.stamp.to_sec(),
            'pos': np.array([pos.x, pos.y, pos.z])
        })

    bag.close()

    print(f"\n读取到 {len(ecef_data)} 条 /fixposition/fpa/odometry (ECEF) 数据")

    if len(ecef_data) < 10:
        print("数据不足!")
        return

    # 第一个点作为原点
    origin_ecef = ecef_data[0]['pos']
    lat, lon = ecef_to_lla(origin_ecef)
    R_ecef_to_enu = get_ecef_to_enu_matrix(lat, lon)

    print(f"\n原点:")
    print(f"  ECEF: ({origin_ecef[0]:.2f}, {origin_ecef[1]:.2f}, {origin_ecef[2]:.2f})")
    print(f"  经纬度: lat={np.degrees(lat):.6f}°, lon={np.degrees(lon):.6f}°")

    # 计算所有点的ENU坐标
    enu_positions = []
    for d in ecef_data:
        delta_ecef = d['pos'] - origin_ecef
        enu = R_ecef_to_enu @ delta_ecef
        enu_positions.append(enu)

    # 总位移
    total_disp_enu = enu_positions[-1] - enu_positions[0]
    print(f"\n总位移 (ENU坐标系):")
    print(f"  dx = {total_disp_enu[0]:.2f}m (东+/西-)")
    print(f"  dy = {total_disp_enu[1]:.2f}m (北+/南-)")
    print(f"  dz = {total_disp_enu[2]:.2f}m (上+/下-)")

    # 计算运动方向
    heading = np.arctan2(total_disp_enu[0], total_disp_enu[1])  # 从北顺时针
    print(f"  运动方向: {np.degrees(heading):.1f}° (从北顺时针, 0°=北, 90°=东, -90°=西)")

    print("\n" + "=" * 70)
    print("测试不同旋转角度:")
    print("=" * 70)

    print("\n假设LIO-SAM中车辆主要向X+方向（前）移动")
    print("需要找到使ENU位移转换为X+方向的旋转角度\n")

    for angle in [0, 90, -90, 180]:
        R = get_yaw_rotation_matrix(np.radians(angle))
        rotated = R @ total_disp_enu
        print(f"  旋转 {angle:4d}°: x={rotated[0]:8.2f}m, y={rotated[1]:8.2f}m, z={rotated[2]:.2f}m")

    # 找最优角度
    print("\n寻找最优角度 (使X方向位移最大化):")
    best_angle = 0
    best_x = -1e10
    for angle in range(-180, 181, 1):
        R = get_yaw_rotation_matrix(np.radians(angle))
        rotated = R @ total_disp_enu
        if rotated[0] > best_x:
            best_x = rotated[0]
            best_angle = angle

    R_best = get_yaw_rotation_matrix(np.radians(best_angle))
    rotated_best = R_best @ total_disp_enu
    print(f"  最优角度: {best_angle}°")
    print(f"  旋转后: x={rotated_best[0]:.2f}m, y={rotated_best[1]:.2f}m")

    print("\n" + "=" * 70)
    print("当前代码设置验证:")
    print("=" * 70)

    current_angle = -90  # 当前代码中的默认值
    R_current = get_yaw_rotation_matrix(np.radians(current_angle))
    rotated_current = R_current @ total_disp_enu
    print(f"\n当前设置 (yaw_offset = {current_angle}°):")
    print(f"  旋转后: x={rotated_current[0]:.2f}m, y={rotated_current[1]:.2f}m")

    if rotated_current[0] > 0 and abs(rotated_current[0]) > abs(rotated_current[1]):
        print("  ✓ 主要是X+方向位移（前进）")
    else:
        print(f"  ✗ 不是X+方向位移，需要调整角度")
        print(f"  建议使用: {best_angle}°")

if __name__ == '__main__':
    import sys
    bag_file = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/info_fixed.bag'
    test_ecef_data(bag_file)
