#!/usr/bin/env python3
"""
测试 fpaOdomConverter 坐标转换的正确性

验证内容：
1. ECEF -> ENU 转换是否正确
2. 90度旋转是否正确应用
3. 转换后的坐标系是否与LIO-SAM对齐
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
    """获取ECEF到ENU的旋转矩阵"""
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    R = np.array([
        [-sin_lon,          cos_lon,          0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
        [cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])
    return R

def get_yaw_rotation_matrix(yaw_deg):
    """获取绕Z轴旋转的矩阵"""
    yaw = np.radians(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def test_coordinate_transform(bag_file):
    """测试坐标转换"""

    print("=" * 70)
    print("坐标转换测试")
    print("=" * 70)

    bag = rosbag.Bag(bag_file, 'r')

    # 读取原始ECEF数据
    ecef_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        ecef_data.append({
            'time': msg.header.stamp.to_sec(),
            'pos': np.array([pos.x, pos.y, pos.z])
        })
        if len(ecef_data) >= 20:
            break

    # 读取转换后的GPS数据（fpaOdomConverter输出）
    # 注意：这需要在运行LIO-SAM时录制

    bag.close()

    if len(ecef_data) < 2:
        print("数据不足!")
        return

    # 使用第一个点作为原点
    origin_ecef = ecef_data[0]['pos']
    lat, lon = ecef_to_lla(origin_ecef)
    R_ecef_to_enu = get_ecef_to_enu_matrix(lat, lon)

    print(f"\n原点 ECEF: {origin_ecef}")
    print(f"原点 经纬度: lat={np.degrees(lat):.6f}°, lon={np.degrees(lon):.6f}°")

    print("\n" + "=" * 70)
    print("测试1: ECEF -> ENU 转换")
    print("=" * 70)

    print("\n前5个点的ENU坐标:")
    enu_positions = []
    for i, d in enumerate(ecef_data[:5]):
        delta_ecef = d['pos'] - origin_ecef
        enu = R_ecef_to_enu @ delta_ecef
        enu_positions.append(enu)
        print(f"  [{i}] ECEF delta: {delta_ecef}")
        print(f"       ENU: x={enu[0]:.4f}m(东), y={enu[1]:.4f}m(北), z={enu[2]:.4f}m(上)")

    print("\n" + "=" * 70)
    print("测试2: 应用-90度旋转 (ENU -> LIO-SAM)")
    print("=" * 70)

    R_yaw = get_yaw_rotation_matrix(-90)  # -90度
    print(f"\n旋转矩阵 R_z(-90°):")
    print(R_yaw)

    print("\n旋转后的坐标 (应该是: X=前/北, Y=左/西):")
    for i, enu in enumerate(enu_positions[:5]):
        rotated = R_yaw @ enu
        print(f"  [{i}] ENU: ({enu[0]:.4f}, {enu[1]:.4f}, {enu[2]:.4f})")
        print(f"       旋转后: x={rotated[0]:.4f}m, y={rotated[1]:.4f}m, z={rotated[2]:.4f}m")

    print("\n" + "=" * 70)
    print("测试3: 验证旋转方向")
    print("=" * 70)

    # 测试向量
    test_vectors = {
        "东方向 (1,0,0)": np.array([1, 0, 0]),
        "北方向 (0,1,0)": np.array([0, 1, 0]),
        "西方向 (-1,0,0)": np.array([-1, 0, 0]),
        "南方向 (0,-1,0)": np.array([0, -1, 0]),
    }

    print("\nENU向量经过-90度旋转后的结果:")
    for name, vec in test_vectors.items():
        rotated = R_yaw @ vec
        print(f"  {name} -> ({rotated[0]:.1f}, {rotated[1]:.1f}, {rotated[2]:.1f})")

    print("""
期望结果 (如果车辆朝北):
  - ENU北方向(0,1,0) 应该变成 LIO-SAM前方向 -> 期望(1,0,0) ✓ 如果旋转正确
  - ENU东方向(1,0,0) 应该变成 LIO-SAM右方向 -> 期望(0,-1,0)
  - ENU西方向(-1,0,0) 应该变成 LIO-SAM左方向 -> 期望(0,1,0)
""")

    # 验证
    north_rotated = R_yaw @ np.array([0, 1, 0])
    if np.allclose(north_rotated, [1, 0, 0]):
        print("✓ 北方向正确转换为前方向")
    else:
        print(f"✗ 北方向转换错误: 期望(1,0,0), 实际{north_rotated}")

    west_rotated = R_yaw @ np.array([-1, 0, 0])
    if np.allclose(west_rotated, [0, 1, 0]):
        print("✓ 西方向正确转换为左方向")
    else:
        print(f"✗ 西方向转换错误: 期望(0,1,0), 实际{west_rotated}")

    print("\n" + "=" * 70)
    print("测试4: 分析实际运动方向")
    print("=" * 70)

    if len(ecef_data) >= 10:
        # 计算一段时间内的位移
        start_enu = R_ecef_to_enu @ (ecef_data[0]['pos'] - origin_ecef)
        end_enu = R_ecef_to_enu @ (ecef_data[-1]['pos'] - origin_ecef)
        displacement_enu = end_enu - start_enu

        print(f"\n位移 (ENU): dx={displacement_enu[0]:.3f}m(东), dy={displacement_enu[1]:.3f}m(北)")

        # 应用旋转
        displacement_rotated = R_yaw @ displacement_enu
        print(f"位移 (旋转后): dx={displacement_rotated[0]:.3f}m, dy={displacement_rotated[1]:.3f}m")

        # 判断主要运动方向
        if abs(displacement_enu[1]) > abs(displacement_enu[0]):
            print(f"\n车辆主要向{'北' if displacement_enu[1] > 0 else '南'}移动")
            print(f"旋转后应该主要是X方向位移 (前进)")
        else:
            print(f"\n车辆主要向{'东' if displacement_enu[0] > 0 else '西'}移动")
            print(f"旋转后应该主要是Y方向位移 (侧向)")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
如果测试3中:
  - 北(0,1,0) -> (1,0,0) ✓
  - 西(-1,0,0) -> (0,1,0) ✓

则旋转矩阵正确。

如果LIO-SAM输出的轨迹形状与GPS轨迹形状一致（只是旋转了90度），
则坐标系对齐正确。
""")

if __name__ == '__main__':
    import sys
    bag_file = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/info_fixed.bag'
    test_coordinate_transform(bag_file)
