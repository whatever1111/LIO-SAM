#!/usr/bin/env python3
"""
验证IMU外参脚本
验证 extrinsicRot 和 extrinsicRPY 配置是否正确

验证方法:
1. extrinsicRot: 检查转换后的加速度重力方向是否正确 (应在-Z方向)
2. extrinsicRPY: 检查转换后的IMU Yaw是否与速度航向一致
3. FpaImu模式: 验证角速度积分四元数的正确性

使用方法:
  rosrun lio_sam verify_imu_extrinsics.py [--fpa]

  --fpa: 使用FpaImu模式，分析CORRIMU数据
"""

import rospy
import math
import numpy as np
import sys
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from collections import deque

# 尝试导入 FpaImu
try:
    from fixposition_driver_msgs.msg import FpaImu
    HAS_FPAIMU = True
except ImportError:
    HAS_FPAIMU = False


def quaternion_to_euler(qx, qy, qz, qw):
    """四元数转欧拉角 (roll, pitch, yaw)"""
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


def normalize_angle(angle):
    """归一化角度到[-180, 180]度"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def apply_rotation(vec, rot_matrix):
    """应用旋转矩阵到向量"""
    return np.dot(rot_matrix, vec)


class QuaternionIntegrator:
    """角速度积分生成四元数"""

    def __init__(self, rot_matrix=None):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_time = None
        self.rot_matrix = rot_matrix if rot_matrix is not None else np.eye(3)

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

    def update(self, gyr, timestamp):
        if self.last_time is None:
            self.last_time = timestamp
            return self.q.copy()

        dt = timestamp - self.last_time
        if dt <= 0 or dt > 1.0:
            self.last_time = timestamp
            return self.q.copy()

        self.last_time = timestamp

        # 应用旋转矩阵变换角速度
        gyr_transformed = self.rot_matrix @ np.array(gyr)

        # 计算角度增量
        omega = np.linalg.norm(gyr_transformed)
        angle = omega * dt

        if omega > 1e-10:
            axis = gyr_transformed / omega
            half_angle = angle / 2
            q_delta = np.array([
                math.cos(half_angle),
                axis[0] * math.sin(half_angle),
                axis[1] * math.sin(half_angle),
                axis[2] * math.sin(half_angle)
            ])
        else:
            q_delta = np.array([1.0, 0.5*gyr_transformed[0]*dt,
                               0.5*gyr_transformed[1]*dt, 0.5*gyr_transformed[2]*dt])
            q_delta = q_delta / np.linalg.norm(q_delta)

        # 四元数乘法: q_new = q_old * q_delta
        self.q = self._quat_multiply(self.q, q_delta)
        self.q = self.q / np.linalg.norm(self.q)
        return self.q.copy()

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


class IMUExtrinsicsVerifier:
    def __init__(self, use_fpa_mode=False):
        rospy.init_node('imu_extrinsics_verifier', anonymous=True)

        self.use_fpa_mode = use_fpa_mode

        # 当前外参配置 (从params.yaml读取或手动设置)
        self.extrinsicRot = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])  # Rz(180°)

        self.extrinsicRPY = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])  # Rz(90°) - 新配置

        # FPA模式下的四元数积分器
        if use_fpa_mode:
            self.integrators = {
                'raw': QuaternionIntegrator(np.eye(3)),
                'extRot': QuaternionIntegrator(self.extrinsicRot),
                'Rz180': QuaternionIntegrator(np.array([[-1,0,0],[0,-1,0],[0,0,1]])),
                'Rz90': QuaternionIntegrator(np.array([[0,-1,0],[1,0,0],[0,0,1]])),
                'Rz-90': QuaternionIntegrator(np.array([[0,1,0],[-1,0,0],[0,0,1]])),
            }

        # 数据缓存
        self.imu_data = deque(maxlen=500)
        self.gps_data = deque(maxlen=500)
        self.corrimu_data = deque(maxlen=500)
        self.integrated_quats = deque(maxlen=500)

        self.last_gps_pos = None
        self.last_gps_time = None

        # 订阅
        if HAS_FPAIMU and use_fpa_mode:
            rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.corrimu_cb)
            print("FPA模式: 订阅 /fixposition/fpa/corrimu")
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb)

        # 定时分析
        rospy.Timer(rospy.Duration(5.0), self.analyze)

        print("=" * 70)
        print("IMU外参验证工具")
        print("=" * 70)
        print("当前外参配置:")
        print(f"  extrinsicRot: {self.extrinsicRot.flatten().tolist()} (Rz(180°))")
        print(f"  extrinsicRPY: {self.extrinsicRPY.flatten().tolist()} (Rz(90°))")
        print()
        print("验证项目:")
        print("  1. extrinsicRot: 转换后重力应在-Z方向 (acc_z ≈ -9.8)")
        print("  2. extrinsicRPY: 转换后IMU Yaw应与速度航向一致")
        print("=" * 70)

    def corrimu_cb(self, msg):
        t = msg.data.header.stamp.to_sec()
        acc = np.array([msg.data.linear_acceleration.x,
                       msg.data.linear_acceleration.y,
                       msg.data.linear_acceleration.z])
        gyr = np.array([msg.data.angular_velocity.x,
                       msg.data.angular_velocity.y,
                       msg.data.angular_velocity.z])

        # 更新所有积分器
        quats = {}
        if self.use_fpa_mode:
            for name, integrator in self.integrators.items():
                quats[name] = integrator.update(gyr, t)

        self.corrimu_data.append({
            'time': t,
            'acc': acc,
            'gyr': gyr,
            'quats': quats,
            'bias_comp': msg.bias_comp,
            'imu_status': msg.imu_status,
        })

    def imu_cb(self, msg):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

        self.imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc': np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z]),
            'gyr': np.array([msg.angular_velocity.x,
                           msg.angular_velocity.y,
                           msg.angular_velocity.z]),
            'quat': np.array([qw, qx, qy, qz]),
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
        })

    def gps_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        vx, vy = 0, 0
        if self.last_gps_pos is not None:
            dt = t - self.last_gps_time
            if dt > 0.01:
                vx = (x - self.last_gps_pos[0]) / dt
                vy = (y - self.last_gps_pos[1]) / dt

        self.last_gps_pos = (x, y)
        self.last_gps_time = t

        self.gps_data.append({
            'time': t, 'x': x, 'y': y, 'vx': vx, 'vy': vy,
        })

    def transform_quaternion_yaw(self, yaw_imu):
        """
        使用extrinsicRPY转换IMU的yaw角
        extrinsicRPY是从IMU坐标系到LiDAR坐标系的旋转
        """
        # 简化处理：对于纯Z轴旋转，yaw的变换就是加上旋转角度
        # extrinsicRPY = Rz(90°) 意味着 yaw_lidar = yaw_imu + 90°
        # 但实际上是 yaw_lidar = yaw_imu - 90° (因为是坐标系变换)

        # 从旋转矩阵提取yaw偏移
        # Rz(theta) = [cos(theta), -sin(theta), 0; sin(theta), cos(theta), 0; 0, 0, 1]
        # extrinsicRPY = [0, -1, 0; 1, 0, 0; 0, 0, 1] => theta = 90°
        rot_yaw = math.atan2(self.extrinsicRPY[1, 0], self.extrinsicRPY[0, 0])

        return yaw_imu + rot_yaw

    def analyze(self, event):
        print("\n" + "=" * 70)
        print(f"数据统计: /imu/data={len(self.imu_data)}, GPS={len(self.gps_data)}, CORRIMU={len(self.corrimu_data)}")
        print("=" * 70)

        # FPA模式特殊分析
        if self.use_fpa_mode and len(self.corrimu_data) > 50:
            self._analyze_fpa_mode()
            return

        if len(self.imu_data) < 50:
            print("等待更多数据...")
            return

        # ============================================
        # 1. 验证 extrinsicRot (加速度/角速度变换)
        # ============================================
        print("\n[1. extrinsicRot 验证] - 加速度坐标变换")

        # 使用原始IMU数据
        acc_raw = np.mean([d['acc'] for d in self.imu_data], axis=0)
        acc_transformed = apply_rotation(acc_raw, self.extrinsicRot)

        print(f"  原始IMU加速度:   X={acc_raw[0]:.3f}, Y={acc_raw[1]:.3f}, Z={acc_raw[2]:.3f} m/s²")
        print(f"  转换后加速度:    X={acc_transformed[0]:.3f}, Y={acc_transformed[1]:.3f}, Z={acc_transformed[2]:.3f} m/s²")

        # 检查重力方向 (LIO-SAM期望重力在-Z方向，即acc_z ≈ -9.8)
        if acc_transformed[2] < -8:
            print(f"  ✓ 重力方向正确: Z={acc_transformed[2]:.2f} m/s² (应 ≈ -9.8)")
        elif acc_transformed[2] > 8:
            print(f"  ✗ 重力方向相反: Z={acc_transformed[2]:.2f} m/s² (应为负值)")
            print(f"    建议: extrinsicRot 需要绕X或Y轴旋转180°")
        else:
            print(f"  ✗ 重力方向错误: Z={acc_transformed[2]:.2f} m/s²")
            print(f"    重力主要在 {'X' if abs(acc_transformed[0]) > 8 else 'Y'} 轴方向")

        # ============================================
        # 2. 验证 extrinsicRPY (姿态变换)
        # ============================================
        print("\n[2. extrinsicRPY 验证] - 姿态/航向变换")

        # 获取速度数据
        valid_gps = [d for d in self.gps_data if math.sqrt(d['vx']**2 + d['vy']**2) > 0.5]

        if not valid_gps:
            print("  速度数据不足，无法验证航向")
            return

        # 计算速度航向
        vx_mean = np.mean([d['vx'] for d in valid_gps])
        vy_mean = np.mean([d['vy'] for d in valid_gps])
        speed = math.sqrt(vx_mean**2 + vy_mean**2)
        vel_heading = math.degrees(math.atan2(vy_mean, vx_mean))

        print(f"  GPS速度: Vx={vx_mean:.2f}, Vy={vy_mean:.2f} m/s (速度={speed:.2f} m/s)")
        print(f"  速度航向: {vel_heading:.1f}°")

        # 获取IMU原始yaw
        imu_yaw_raw = math.degrees(np.mean([d['yaw'] for d in self.imu_data]))

        # 应用extrinsicRPY变换
        imu_yaw_transformed = math.degrees(self.transform_quaternion_yaw(math.radians(imu_yaw_raw)))
        imu_yaw_transformed = normalize_angle(imu_yaw_transformed)

        print(f"\n  IMU原始Yaw: {imu_yaw_raw:.1f}°")
        print(f"  转换后Yaw:  {imu_yaw_transformed:.1f}°")

        # 计算与速度航向的差异
        yaw_diff = normalize_angle(imu_yaw_transformed - vel_heading)

        print(f"\n  转换后Yaw vs 速度航向差: {yaw_diff:.1f}°")

        if abs(yaw_diff) < 20:
            print(f"  ✓ 航向一致! extrinsicRPY 设置正确")
        elif abs(abs(yaw_diff) - 180) < 20:
            print(f"  ! 航向相差180°，可能是倒车或需要调整")
        elif abs(yaw_diff - 90) < 20:
            print(f"  ✗ 航向差90°，extrinsicRPY 需要额外 Rz(-90°)")
        elif abs(yaw_diff + 90) < 20:
            print(f"  ✗ 航向差-90°，extrinsicRPY 需要额外 Rz(+90°)")
        else:
            print(f"  △ 航向差异 {yaw_diff:.1f}° 不是典型值")

        # ============================================
        # 3. 计算建议的外参
        # ============================================
        print("\n" + "=" * 70)
        print("[外参计算]")
        print("=" * 70)

        # --- extrinsicRot 计算 ---
        # 目标: 转换后 acc_z 应该是负值 (重力向-Z)
        print("\n  [extrinsicRot 计算]")
        print(f"  原始重力方向: Z={acc_raw[2]:.2f} m/s²")

        if acc_raw[2] > 8:
            # 重力在+Z，需要翻转Z轴
            # Rz(180°) × Rx(180°) = [-1,0,0; 0,1,0; 0,0,-1]
            suggested_rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            rot_desc = "Rz(180°)×Rx(180°) = [-1,0,0, 0,1,0, 0,0,-1]"
        elif acc_raw[2] < -8:
            # 重力已经在-Z，可能只需要XY平面旋转
            suggested_rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            rot_desc = "Rz(180°) = [-1,0,0, 0,-1,0, 0,0,1]"
        else:
            suggested_rot = self.extrinsicRot
            rot_desc = "需要手动分析"

        acc_after_suggested = apply_rotation(acc_raw, suggested_rot)
        print(f"  建议 extrinsicRot: {rot_desc}")
        print(f"  应用后加速度: Z={acc_after_suggested[2]:.2f} m/s²")

        # --- extrinsicRPY 计算 ---
        print("\n  [extrinsicRPY 计算]")
        print(f"  IMU原始Yaw: {imu_yaw_raw:.1f}°")
        print(f"  速度航向:   {vel_heading:.1f}°")

        # 需要的yaw偏移 = 速度航向 - IMU原始Yaw
        needed_yaw_offset = normalize_angle(vel_heading - imu_yaw_raw)
        print(f"  需要的Yaw偏移: {needed_yaw_offset:.1f}°")

        # 将偏移量圆整到最近的90°
        rounded_offset = round(needed_yaw_offset / 90) * 90
        rounded_offset = normalize_angle(rounded_offset)
        print(f"  圆整到90°倍数: {rounded_offset:.0f}°")

        # 生成对应的旋转矩阵
        theta = math.radians(rounded_offset)
        suggested_rpy = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])

        # 应用后的yaw
        yaw_after_suggested = normalize_angle(imu_yaw_raw + rounded_offset)
        final_diff = normalize_angle(yaw_after_suggested - vel_heading)

        print(f"  应用后Yaw: {yaw_after_suggested:.1f}°")
        print(f"  与速度航向差: {final_diff:.1f}°")

        # ============================================
        # 4. 综合结论
        # ============================================
        print("\n" + "=" * 70)
        print("[综合结论]")
        print("=" * 70)

        rot_ok = acc_after_suggested[2] < -8
        rpy_ok = abs(final_diff) < 25

        print(f"  extrinsicRot: {'✓ 正确' if rot_ok else '✗ 需要调整'}")
        print(f"  extrinsicRPY: {'✓ 正确' if rpy_ok else '△ 可能需要微调'}")

        # 生成建议配置
        print("\n  [建议配置] (复制到 params.yaml)")
        print("  extrinsicRot: [{}, {}, {},".format(
            int(suggested_rot[0,0]), int(suggested_rot[0,1]), int(suggested_rot[0,2])))
        print("                {}, {}, {},".format(
            int(suggested_rot[1,0]), int(suggested_rot[1,1]), int(suggested_rot[1,2])))
        print("                {}, {}, {}]".format(
            int(suggested_rot[2,0]), int(suggested_rot[2,1]), int(suggested_rot[2,2])))

        print("  extrinsicRPY: [{:.0f}, {:.0f}, {:.0f},".format(
            suggested_rpy[0,0], suggested_rpy[0,1], suggested_rpy[0,2]))
        print("                {:.0f}, {:.0f}, {:.0f},".format(
            suggested_rpy[1,0], suggested_rpy[1,1], suggested_rpy[1,2]))
        print("                {:.0f}, {:.0f}, {:.0f}]".format(
            suggested_rpy[2,0], suggested_rpy[2,1], suggested_rpy[2,2]))

        if rounded_offset == 0:
            rpy_name = "单位矩阵"
        elif rounded_offset == 90:
            rpy_name = "Rz(90°)"
        elif rounded_offset == -90 or rounded_offset == 270:
            rpy_name = "Rz(-90°)"
        elif abs(rounded_offset) == 180:
            rpy_name = "Rz(180°)"
        else:
            rpy_name = f"Rz({rounded_offset}°)"

        print(f"\n  extrinsicRPY 含义: {rpy_name}")


    def _analyze_fpa_mode(self):
        """FPA模式的四元数积分验证"""
        print("\n[FPA模式 - CORRIMU四元数积分验证]")

        # ============================================
        # 1. 验证加速度重力方向
        # ============================================
        print("\n[1. 加速度重力方向验证]")

        acc_raw_mean = np.mean([d['acc'] for d in self.corrimu_data], axis=0)
        print(f"  CORRIMU原始加速度: X={acc_raw_mean[0]:.3f}, Y={acc_raw_mean[1]:.3f}, Z={acc_raw_mean[2]:.3f} m/s²")

        # 测试不同旋转矩阵
        test_rotations = {
            'Identity': np.eye(3),
            'Rz(180°)': np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
            'Rz(90°)': np.array([[0,-1,0],[1,0,0],[0,0,1]]),
            'Rz(-90°)': np.array([[0,1,0],[-1,0,0],[0,0,1]]),
        }

        print("\n  不同旋转矩阵应用后的结果:")
        gravity_check = {}
        for name, rot in test_rotations.items():
            acc_rot = rot @ acc_raw_mean
            gravity_ok = acc_rot[2] < -8
            gravity_check[name] = (gravity_ok, acc_rot[2])
            status = "✓" if gravity_ok else "✗"
            print(f"    {name:12s}: Z={acc_rot[2]:+.2f} m/s² {status}")

        # ============================================
        # 2. 验证四元数积分 vs GPS航向
        # ============================================
        print("\n[2. 四元数积分 vs GPS航向]")

        valid_gps = [d for d in self.gps_data if math.sqrt(d['vx']**2 + d['vy']**2) > 0.5]
        if len(valid_gps) < 10:
            print("  GPS速度数据不足，跳过航向验证")
            gps_heading = None
        else:
            vx_mean = np.mean([d['vx'] for d in valid_gps[-50:]])
            vy_mean = np.mean([d['vy'] for d in valid_gps[-50:]])
            speed = math.sqrt(vx_mean**2 + vy_mean**2)
            gps_heading = math.degrees(math.atan2(vy_mean, vx_mean))

            print(f"  GPS速度: Vx={vx_mean:.2f}, Vy={vy_mean:.2f} m/s (速度={speed:.2f} m/s)")
            print(f"  GPS航向: {gps_heading:.1f}°")

            # 获取最新的积分四元数
            latest = self.corrimu_data[-1]

            if latest['quats']:
                print("\n  不同积分配置的Yaw角:")
                yaw_results = {}
                for name, q in latest['quats'].items():
                    roll, pitch, yaw = quaternion_to_euler(q[1], q[2], q[3], q[0])  # x,y,z,w
                    yaw_deg = math.degrees(yaw)
                    diff = normalize_angle(yaw_deg - gps_heading)
                    yaw_results[name] = {'yaw': yaw_deg, 'diff': diff}

                    status = "✓" if abs(diff) < 30 else ("~" if abs(normalize_angle(abs(diff) - 180)) < 30 else "✗")
                    print(f"    {name:10s}: Yaw={yaw_deg:+7.1f}°, 与GPS差={diff:+7.1f}° {status}")

        # ============================================
        # 3. 外参建议
        # ============================================
        print("\n" + "=" * 70)
        print("[外参配置建议]")
        print("=" * 70)

        # extrinsicRot建议
        print("\n  [extrinsicRot - 加速度/角速度变换]")
        best_rot = None
        for name, (ok, z_val) in gravity_check.items():
            if ok:
                best_rot = name
                print(f"    建议使用: {name} (Z={z_val:.2f})")
                if name == 'Identity':
                    print("    参数: [1,0,0, 0,1,0, 0,0,1]")
                elif name == 'Rz(180°)':
                    print("    参数: [-1,0,0, 0,-1,0, 0,0,1]")
                elif name == 'Rz(90°)':
                    print("    参数: [0,-1,0, 1,0,0, 0,0,1]")
                elif name == 'Rz(-90°)':
                    print("    参数: [0,1,0, -1,0,0, 0,0,1]")
                break

        if not best_rot:
            print("    警告: 没有找到使重力在-Z方向的旋转矩阵")

        # extrinsicRPY建议 (对于FPA，这决定了积分后四元数的yaw偏移)
        print("\n  [extrinsicRPY - 姿态变换]")
        print("    注意: 对于FpaImu/CORRIMU，没有原生四元数")
        print("    角速度积分后的四元数已经在extRot变换后的坐标系中")

        if gps_heading is not None and 'extRot' in yaw_results:
            needed_offset = normalize_angle(gps_heading - yaw_results['extRot']['yaw'])
            rounded_offset = round(needed_offset / 90) * 90
            print(f"    基于extRot积分，需要Yaw偏移: {needed_offset:.1f}° (圆整: {rounded_offset:.0f}°)")

            if rounded_offset == 0:
                print("    建议 extrinsicRPY: Identity = [1,0,0, 0,1,0, 0,0,1]")
            elif rounded_offset == 90:
                print("    建议 extrinsicRPY: Rz(90°) = [0,-1,0, 1,0,0, 0,0,1]")
            elif rounded_offset == -90 or rounded_offset == 270:
                print("    建议 extrinsicRPY: Rz(-90°) = [0,1,0, -1,0,0, 0,0,1]")
            elif abs(rounded_offset) == 180:
                print("    建议 extrinsicRPY: Rz(180°) = [-1,0,0, 0,-1,0, 0,0,1]")

        # CORRIMU状态
        if self.corrimu_data:
            latest = self.corrimu_data[-1]
            print(f"\n  [CORRIMU状态]")
            print(f"    bias_comp: {latest.get('bias_comp', 'N/A')}")
            print(f"    imu_status: {latest.get('imu_status', 'N/A')}")


def main():
    # 检查命令行参数
    use_fpa = '--fpa' in sys.argv

    if use_fpa and not HAS_FPAIMU:
        print("ERROR: --fpa 模式需要 fixposition_driver_msgs")
        print("请确保已source包含该消息的工作空间")
        return

    verifier = IMUExtrinsicsVerifier(use_fpa_mode=use_fpa)
    rospy.spin()


if __name__ == '__main__':
    main()
