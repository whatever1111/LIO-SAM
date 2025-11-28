#!/usr/bin/env python3
"""
测试不同旋转角度的效果，找出正确的yaw_offset
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

def test_rotation_angles(bag_file):
    print("=" * 70)
    print("测试不同旋转角度")
    print("=" * 70)

    bag = rosbag.Bag(bag_file, 'r')

    ecef_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        ecef_data.append(np.array([pos.x, pos.y, pos.z]))

    enu_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
        pos = msg.pose.pose.position
        enu_data.append(np.array([pos.x, pos.y, pos.z]))

    bag.close()

    # 原始ENU位移（odomenu提供的）
    if len(enu_data) > 0:
        enu_disp = enu_data[-1] - enu_data[0]
        print(f"\n原始ENU位移 (odomenu):")
        print(f"  dx={enu_disp[0]:.2f}m (东), dy={enu_disp[1]:.2f}m (北), dz={enu_disp[2]:.2f}m (上)")

        # 计算运动方向角
        heading_rad = np.arctan2(enu_disp[0], enu_disp[1])  # 从北开始顺时针
        print(f"  运动方向: {np.degrees(heading_rad):.1f}° (从北顺时针)")

    # 我们从ECEF计算的ENU
    origin = ecef_data[0]
    lat, lon = ecef_to_lla(origin)
    R_ecef_to_enu = get_ecef_to_enu_matrix(lat, lon)

    enu_calc_start = R_ecef_to_enu @ (ecef_data[0] - origin)
    enu_calc_end = R_ecef_to_enu @ (ecef_data[-1] - origin)
    enu_calc_disp = enu_calc_end - enu_calc_start

    print(f"\n我们计算的ENU位移:")
    print(f"  dx={enu_calc_disp[0]:.2f}m (东), dy={enu_calc_disp[1]:.2f}m (北), dz={enu_calc_disp[2]:.2f}m (上)")

    print("\n" + "=" * 70)
    print("测试不同旋转角度的效果:")
    print("=" * 70)

    angles_to_test = [0, 90, -90, 180, 45, -45]

    for angle in angles_to_test:
        R = get_yaw_rotation_matrix(np.radians(angle))
        rotated = R @ enu_calc_disp

        # 判断主要方向
        if abs(rotated[0]) > abs(rotated[1]):
            main_dir = "X (前后)" if rotated[0] > 0 else "X (后)"
        else:
            main_dir = "Y (左)" if rotated[1] > 0 else "Y (右)"

        print(f"\n  角度 {angle:4d}°: x={rotated[0]:8.2f}m, y={rotated[1]:8.2f}m -> 主要方向: {main_dir}")

    print("\n" + "=" * 70)
    print("分析:")
    print("=" * 70)
    print("""
LIO-SAM坐标系: X=前, Y=左, Z=上
期望: 如果车辆向前行驶，应该主要是X方向正位移

根据上面的测试结果:
- 选择使X方向位移最大（正值）的旋转角度
- 或者让位移方向与LIO-SAM的X轴对齐
""")

    # 计算最优旋转角度（使X方向位移最大化）
    print("\n寻找最优旋转角度 (使X+方向位移最大):")
    best_angle = 0
    best_x = -1e10

    for angle in range(-180, 181, 5):
        R = get_yaw_rotation_matrix(np.radians(angle))
        rotated = R @ enu_calc_disp
        if rotated[0] > best_x:
            best_x = rotated[0]
            best_angle = angle

    print(f"  最优角度: {best_angle}°")
    R_best = get_yaw_rotation_matrix(np.radians(best_angle))
    rotated_best = R_best @ enu_calc_disp
    print(f"  旋转后位移: x={rotated_best[0]:.2f}m, y={rotated_best[1]:.2f}m")

if __name__ == '__main__':
    import sys
    bag_file = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/info_fixed.bag'
    test_rotation_angles(bag_file)
