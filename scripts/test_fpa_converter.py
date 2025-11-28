#!/usr/bin/env python3
"""
完整测试：验证fpaOdomConverter的坐标转换是否与代码实现一致
"""

import numpy as np
from scipy.spatial.transform import Rotation
import rosbag

def ecef_to_lla(ecef):
    """ECEF转经纬度"""
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
    """获取ECEF到ENU的旋转矩阵 - 与代码一致"""
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    R = np.array([
        [-sin_lon,          cos_lon,          0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
        [cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])
    return R

def get_yaw_rotation_matrix(yaw_rad):
    """获取绕Z轴旋转的矩阵 - 与代码一致"""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    # 代码中: R << cos_yaw, -sin_yaw, 0,
    #              sin_yaw,  cos_yaw, 0,
    #              0,        0,       1;
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def simulate_fpa_converter(ecef_positions, yaw_offset_deg=-90):
    """模拟fpaOdomConverter的转换过程"""

    if len(ecef_positions) == 0:
        return []

    # 第一个点作为原点
    origin_ecef = ecef_positions[0]
    lat, lon = ecef_to_lla(origin_ecef)

    R_ecef_to_enu = get_ecef_to_enu_matrix(lat, lon)
    R_enu_to_liosam = get_yaw_rotation_matrix(np.radians(yaw_offset_deg))

    converted = []
    for i, ecef in enumerate(ecef_positions):
        if i == 0:
            converted.append(np.array([0, 0, 0]))  # 第一个点是原点
        else:
            delta_ecef = ecef - origin_ecef
            enu = R_ecef_to_enu @ delta_ecef
            aligned = R_enu_to_liosam @ enu
            converted.append(aligned)

    return converted, R_ecef_to_enu, R_enu_to_liosam

def test_with_bag(bag_file):
    """使用bag文件测试"""

    print("=" * 70)
    print("fpaOdomConverter 转换验证测试")
    print("=" * 70)

    bag = rosbag.Bag(bag_file, 'r')

    # 读取所有ECEF数据
    ecef_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        ecef_data.append({
            'time': msg.header.stamp.to_sec(),
            'pos': np.array([pos.x, pos.y, pos.z])
        })

    # 读取odomenu数据作为参考
    enu_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odomenu']):
        pos = msg.pose.pose.position
        enu_data.append({
            'time': msg.header.stamp.to_sec(),
            'pos': np.array([pos.x, pos.y, pos.z])
        })

    bag.close()

    print(f"\n读取到 {len(ecef_data)} 条ECEF数据")
    print(f"读取到 {len(enu_data)} 条ENU数据")

    if len(ecef_data) < 10:
        print("数据不足!")
        return

    # 模拟转换
    ecef_positions = [d['pos'] for d in ecef_data]
    converted, R_ecef_to_enu, R_enu_to_liosam = simulate_fpa_converter(ecef_positions, -90)

    print("\n" + "=" * 70)
    print("1. 验证旋转矩阵")
    print("=" * 70)

    print("\nR_enu_to_liosam (绕Z轴旋转-90度):")
    print(R_enu_to_liosam)

    # 测试关键向量
    tests = [
        ("ENU东(1,0,0)", np.array([1,0,0]), "应变为(0,-1,0)右"),
        ("ENU北(0,1,0)", np.array([0,1,0]), "应变为(1,0,0)前"),
        ("ENU西(-1,0,0)", np.array([-1,0,0]), "应变为(0,1,0)左"),
    ]

    print("\n关键向量转换测试:")
    for name, vec, expected in tests:
        result = R_enu_to_liosam @ vec
        print(f"  {name} -> ({result[0]:.2f}, {result[1]:.2f}, {result[2]:.2f}) {expected}")

    print("\n" + "=" * 70)
    print("2. 与odomenu数据对比 (验证ECEF->ENU)")
    print("=" * 70)

    if len(enu_data) > 0:
        # odomenu的原点
        enu_origin = enu_data[0]['pos']
        print(f"\nodomenu第一个点: ({enu_origin[0]:.3f}, {enu_origin[1]:.3f}, {enu_origin[2]:.3f})")
        print("注意: odomenu可能有自己的原点偏移")

        # 计算ENU位移
        if len(enu_data) > 100:
            enu_displacement = enu_data[100]['pos'] - enu_data[0]['pos']
            print(f"\nodomenu前100帧位移: dx={enu_displacement[0]:.3f}(东), dy={enu_displacement[1]:.3f}(北), dz={enu_displacement[2]:.3f}(上)")

            # 我们的转换结果（不含90度旋转）
            origin_ecef = ecef_data[0]['pos']
            lat, lon = ecef_to_lla(origin_ecef)
            R = get_ecef_to_enu_matrix(lat, lon)

            our_enu_0 = R @ (ecef_data[0]['pos'] - origin_ecef)
            our_enu_100 = R @ (ecef_data[100]['pos'] - origin_ecef)
            our_displacement = our_enu_100 - our_enu_0

            print(f"我们计算的ENU位移: dx={our_displacement[0]:.3f}(东), dy={our_displacement[1]:.3f}(北), dz={our_displacement[2]:.3f}(上)")

            diff = np.linalg.norm(enu_displacement - our_displacement)
            print(f"差异: {diff:.4f}m")

            if diff < 0.1:
                print("✓ ECEF->ENU转换正确")
            else:
                print("✗ ECEF->ENU转换可能有问题")

    print("\n" + "=" * 70)
    print("3. 分析整体轨迹")
    print("=" * 70)

    # 找出有显著位移的区间
    total_points = len(converted)
    sample_indices = [0, total_points//4, total_points//2, 3*total_points//4, total_points-1]

    print("\n转换后的轨迹采样点:")
    for i in sample_indices:
        if i < len(converted):
            p = converted[i]
            print(f"  [{i:5d}] x={p[0]:10.3f}m, y={p[1]:10.3f}m, z={p[2]:10.3f}m")

    # 计算总位移
    start = converted[0]
    end = converted[-1]
    total_disp = end - start
    print(f"\n总位移: dx={total_disp[0]:.3f}m, dy={total_disp[1]:.3f}m, dz={total_disp[2]:.3f}m")
    print(f"总距离: {np.linalg.norm(total_disp):.3f}m")

    # 分析主要运动方向
    if abs(total_disp[0]) > abs(total_disp[1]):
        print(f"\n主要运动方向: X轴 ({'前进' if total_disp[0] > 0 else '后退'})")
    else:
        print(f"\n主要运动方向: Y轴 ({'左' if total_disp[1] > 0 else '右'})")

    print("\n" + "=" * 70)
    print("4. 结论")
    print("=" * 70)
    print("""
验证要点:
1. 旋转矩阵R_z(-90°)应使: 北->前(X+), 东->右(Y-), 西->左(Y+)
2. 如果车辆主要朝北行驶，转换后应主要是X方向位移
3. 如果车辆主要朝东行驶，转换后应主要是Y方向负位移

如果实际运行时LIO-SAM输出与GPS仍有偏差，可能原因:
- GPS和LiDAR第一帧时间不同步（原点不一致）
- IMU初始航向与假设的北向不一致
- 需要调整yaw_offset参数
""")

if __name__ == '__main__':
    import sys
    bag_file = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/info_fixed.bag'
    test_with_bag(bag_file)
