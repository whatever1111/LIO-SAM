#!/usr/bin/env python3
"""
验证外参矩阵对IMU数据的影响
"""

import numpy as np

print("="*70)
print("外参矩阵分析")
print("="*70)

# 当前配置
extRot = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

extRPY = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

print("\n1. extRot (用于加速度/角速度):")
print(extRot)
print(f"   行列式: {np.linalg.det(extRot):.1f}")

print("\n2. extRPY (用于姿态):")
print(extRPY)
print(f"   行列式: {np.linalg.det(extRPY):.1f}")

# 检查是否相同
if np.allclose(extRot, extRPY):
    print("\n✅ 两个矩阵一致")
else:
    print("\n❌ 两个矩阵不一致!")
    print("   这会导致IMU数据转换不一致")

# 模拟IMU数据转换
print("\n" + "="*70)
print("模拟IMU数据转换")
print("="*70)

# 假设IMU重力在Z轴 (静止时加速度 = [0, 0, 9.8])
imu_acc = np.array([0, 0, 9.8])
print(f"\n原始IMU加速度: {imu_acc}")

converted_acc = extRot @ imu_acc
print(f"extRot转换后:   {converted_acc}")
print(f"重力现在在: {'Z' if abs(converted_acc[2]) > 5 else 'X' if abs(converted_acc[0]) > 5 else 'Y'}轴")

# 用户实际IMU数据 (从之前的分析)
actual_imu = np.array([-0.04, -0.20, 9.84])
print(f"\n实际IMU数据:    {actual_imu}")
converted_actual = extRot @ actual_imu
print(f"extRot转换后:   {converted_actual}")

# 期望: LIO-SAM期望重力在Z轴正方向
print("\n" + "="*70)
print("LIO-SAM期望")
print("="*70)
print("LIO-SAM期望转换后重力在Z轴 (向下为正)")
print(f"当前转换后Z值: {converted_actual[2]:.2f} m/s²")

if converted_actual[2] < 0:
    print("⚠️  重力在-Z方向,这可能是正确的(取决于IMU安装)")
elif converted_actual[2] > 9:
    print("⚠️  重力在+Z方向,Z轴向下")

# 姿态转换对比
print("\n" + "="*70)
print("姿态转换分析")
print("="*70)

# 单位四元数 (无旋转)
from scipy.spatial.transform import Rotation as R

# extRPY的逆用于姿态
extRPY_rot = R.from_matrix(extRPY)
extQRPY = extRPY_rot.inv()

print(f"extRPY欧拉角: {extRPY_rot.as_euler('xyz', degrees=True)}")
print(f"extQRPY欧拉角: {extQRPY.as_euler('xyz', degrees=True)}")

# 如果有初始IMU姿态
# 用户IMU: roll=-1.14°, pitch=0.28°, yaw=84.20°
imu_rot = R.from_euler('xyz', [-1.14, 0.28, 84.20], degrees=True)
final_rot = imu_rot * extQRPY

print(f"\n初始IMU姿态: roll=-1.14°, pitch=0.28°, yaw=84.20°")
print(f"extQRPY转换后: {final_rot.as_euler('xyz', degrees=True)}")

print("\n" + "="*70)
print("结论")
print("="*70)
print("extRot 和 extRPY 不一致会导致:")
print("1. 加速度方向与姿态不匹配")
print("2. 重力补偿错误")
print("3. deskew失败")
print("4. 优化崩溃")
print("\n建议: 确保两个矩阵表示相同的旋转!")
