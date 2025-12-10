#!/usr/bin/env python3
"""
FpaImu 四元数计算验证脚本

验证内容:
1. CORRIMU 的角速度积分生成的四元数是否正确
2. extrinsicRot 应用于加速度/角速度的效果
3. 验证 FPA 下的正确外参配置

验证方法:
- 比较积分四元数与GPS速度航向
- 检查重力方向是否正确（应在-Z轴）
- 验证IMU yaw与GPS航向的一致性
"""

import rospy
import math
import numpy as np
from collections import deque
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

# 尝试导入 FpaImu
try:
    from fixposition_driver_msgs.msg import FpaImu
    HAS_FPAIMU = True
except ImportError:
    HAS_FPAIMU = False
    print("WARNING: fixposition_driver_msgs not available")


class QuaternionIntegrator:
    """角速度积分生成四元数"""

    def __init__(self, rot_matrix=None):
        """
        Args:
            rot_matrix: 3x3旋转矩阵，用于变换角速度
        """
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_time = None
        self.rot_matrix = rot_matrix if rot_matrix is not None else np.eye(3)

    def reset(self, q=None):
        if q is not None:
            self.q = q.copy()
        else:
            self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_time = None

    def update(self, gyr, timestamp):
        """
        使用角速度更新四元数

        Args:
            gyr: [wx, wy, wz] 角速度 (rad/s)
            timestamp: 时间戳 (秒)
        """
        if self.last_time is None:
            self.last_time = timestamp
            return self.q.copy()

        dt = timestamp - self.last_time
        if dt <= 0 or dt > 1.0:
            self.last_time = timestamp
            return self.q.copy()

        self.last_time = timestamp

        # 应用旋转矩阵变换角速度
        gyr_vec = np.array(gyr)
        gyr_transformed = self.rot_matrix @ gyr_vec

        # 计算角度增量
        omega = np.linalg.norm(gyr_transformed)
        angle = omega * dt

        if omega > 1e-10:
            axis = gyr_transformed / omega
            # 轴角到四元数
            half_angle = angle / 2
            q_delta = np.array([
                math.cos(half_angle),
                axis[0] * math.sin(half_angle),
                axis[1] * math.sin(half_angle),
                axis[2] * math.sin(half_angle)
            ])
        else:
            # 小角度近似
            q_delta = np.array([1.0, 0.5*gyr_transformed[0]*dt,
                               0.5*gyr_transformed[1]*dt, 0.5*gyr_transformed[2]*dt])
            q_delta = q_delta / np.linalg.norm(q_delta)

        # 四元数乘法: q_new = q_old * q_delta (body frame)
        self.q = self._quat_multiply(self.q, q_delta)
        self.q = self.q / np.linalg.norm(self.q)

        return self.q.copy()

    def _quat_multiply(self, q1, q2):
        """四元数乘法 [w,x,y,z] format"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])


def quaternion_to_euler(q):
    """四元数转欧拉角 [w,x,y,z] -> [roll, pitch, yaw]"""
    w, x, y, z = q

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


def normalize_angle(angle_deg):
    """归一化角度到[-180, 180]度"""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    return angle_deg


class FpaImuQuaternionVerifier:
    def __init__(self):
        rospy.init_node('fpaimu_quaternion_verifier', anonymous=True)

        # 当前外参配置 (从params.yaml)
        # extrinsicRot: 用于变换加速度和角速度
        self.extrinsicRot = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])  # Rz(180°)

        # extrinsicRPY: 用于变换四元数姿态
        self.extrinsicRPY = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])  # Rz(90°)

        # 创建多个四元数积分器测试不同配置
        self.integrators = {
            'raw': QuaternionIntegrator(np.eye(3)),  # 原始数据，不变换
            'extRot': QuaternionIntegrator(self.extrinsicRot),  # 只用extRot
            'extRPY': QuaternionIntegrator(self.extrinsicRPY),  # 只用extRPY (当前配置)
            'Rz180': QuaternionIntegrator(np.array([[-1,0,0],[0,-1,0],[0,0,1]])),  # Rz(180)
            'Rz90': QuaternionIntegrator(np.array([[0,-1,0],[1,0,0],[0,0,1]])),   # Rz(90)
            'Rz-90': QuaternionIntegrator(np.array([[0,1,0],[-1,0,0],[0,0,1]])),  # Rz(-90)
        }

        # 数据缓存
        self.corrimu_data = deque(maxlen=2000)
        self.imu_data = deque(maxlen=2000)
        self.gps_data = deque(maxlen=500)
        self.lio_imu_data = deque(maxlen=2000)  # LIO-SAM发布的带四元数的IMU

        self.last_gps_pos = None
        self.last_gps_time = None

        # 订阅
        if HAS_FPAIMU:
            rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.corrimu_cb)
            print("Subscribed to /fixposition/fpa/corrimu (FpaImu)")

        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/lio_sam/imu/data', Imu, self.lio_imu_cb)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb)

        # 定时分析
        rospy.Timer(rospy.Duration(5.0), self.analyze)

        self._print_header()

    def _print_header(self):
        print("=" * 80)
        print("FpaImu 四元数计算验证工具")
        print("=" * 80)
        print("\n验证目标:")
        print("  1. 检查 CORRIMU 角速度积分产生的四元数是否正确")
        print("  2. 验证不同 extrinsicRot 配置下的效果")
        print("  3. 确定 FPA 下的正确外参")
        print("\n当前配置:")
        print(f"  extrinsicRot: Rz(180°) = {self.extrinsicRot.flatten().tolist()}")
        print(f"  extrinsicRPY: Rz(90°)  = {self.extrinsicRPY.flatten().tolist()}")
        print("\n数据源:")
        print("  - /fixposition/fpa/corrimu: CORRIMU原始数据")
        print("  - /imu/data: 标准IMU数据(���果有)")
        print("  - /lio_sam/imu/data: LIO-SAM积分后的IMU")
        print("  - /odometry/gps: GPS里程计")
        print("=" * 80)

    def corrimu_cb(self, msg):
        """处理CORRIMU消息"""
        t = msg.data.header.stamp.to_sec()

        # 原始数据
        acc_raw = np.array([msg.data.linear_acceleration.x,
                           msg.data.linear_acceleration.y,
                           msg.data.linear_acceleration.z])
        gyr_raw = np.array([msg.data.angular_velocity.x,
                           msg.data.angular_velocity.y,
                           msg.data.angular_velocity.z])

        # 更新所有积分器
        quats = {}
        for name, integrator in self.integrators.items():
            quats[name] = integrator.update(gyr_raw, t)

        self.corrimu_data.append({
            'time': t,
            'acc_raw': acc_raw,
            'gyr_raw': gyr_raw,
            'quats': quats,
            'bias_comp': msg.bias_comp,
            'imu_status': msg.imu_status,
        })

    def imu_cb(self, msg):
        """处理标准IMU消息"""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quaternion_to_euler([qw, qx, qy, qz])

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

    def lio_imu_cb(self, msg):
        """处理LIO-SAM发布的带四元数的IMU"""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quaternion_to_euler([qw, qx, qy, qz])

        self.lio_imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'quat': np.array([qw, qx, qy, qz]),
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'cov': msg.orientation_covariance[0],  # -1表示无效
        })

    def gps_cb(self, msg):
        """处理GPS里程计"""
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

    def analyze(self, event):
        print("\n" + "=" * 80)
        print(f"数据统计: CORRIMU={len(self.corrimu_data)}, /imu/data={len(self.imu_data)}, "
              f"LIO-IMU={len(self.lio_imu_data)}, GPS={len(self.gps_data)}")
        print("=" * 80)

        if len(self.corrimu_data) < 100:
            print("等待CORRIMU数据...")
            return

        # ============================================
        # 1. 验证加速度重力方向
        # ============================================
        print("\n[1. 加速度重力方向验证]")

        acc_raw_mean = np.mean([d['acc_raw'] for d in self.corrimu_data], axis=0)
        print(f"  CORRIMU原始加速度: X={acc_raw_mean[0]:.3f}, Y={acc_raw_mean[1]:.3f}, Z={acc_raw_mean[2]:.3f} m/s²")

        # 测试不同旋转矩阵
        test_rotations = {
            'Identity': np.eye(3),
            'Rz(180°)': np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
            'Rz(90°)': np.array([[0,-1,0],[1,0,0],[0,0,1]]),
            'Rz(-90°)': np.array([[0,1,0],[-1,0,0],[0,0,1]]),
            'Rx(180°)': np.array([[1,0,0],[0,-1,0],[0,0,-1]]),
            'Ry(180°)': np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
        }

        print("\n  不同旋转矩阵应用后的结果:")
        gravity_check = {}
        for name, rot in test_rotations.items():
            acc_rot = rot @ acc_raw_mean
            gravity_ok = acc_rot[2] < -8
            gravity_check[name] = gravity_ok
            status = "✓" if gravity_ok else "✗"
            print(f"    {name:12s}: Z={acc_rot[2]:+.2f} m/s² {status}")

        # ============================================
        # 2. 验证四元数积分 vs GPS航向
        # ============================================
        print("\n[2. 四元数积分 vs GPS航向]")

        # 获取有效速度数据
        valid_gps = [d for d in self.gps_data if math.sqrt(d['vx']**2 + d['vy']**2) > 1.0]

        if len(valid_gps) < 10:
            print("  GPS速度数据不足，跳过航向验证")
        else:
            # 计算GPS航向
            vx_mean = np.mean([d['vx'] for d in valid_gps[-50:]])
            vy_mean = np.mean([d['vy'] for d in valid_gps[-50:]])
            speed = math.sqrt(vx_mean**2 + vy_mean**2)
            gps_heading = math.degrees(math.atan2(vy_mean, vx_mean))

            print(f"  GPS速度: Vx={vx_mean:.2f}, Vy={vy_mean:.2f} m/s (速度={speed:.2f} m/s)")
            print(f"  GPS航向: {gps_heading:.1f}°")

            # 获取最新的积分四元数
            latest = self.corrimu_data[-1]

            print("\n  不同积分配置的Yaw角:")
            yaw_results = {}
            for name, q in latest['quats'].items():
                roll, pitch, yaw = quaternion_to_euler(q)
                yaw_deg = math.degrees(yaw)
                diff = normalize_angle(yaw_deg - gps_heading)
                yaw_results[name] = {'yaw': yaw_deg, 'diff': diff}

                status = "✓" if abs(diff) < 30 else ("~" if abs(normalize_angle(abs(diff) - 180)) < 30 else "✗")
                print(f"    {name:10s}: Yaw={yaw_deg:+7.1f}°, 与GPS差={diff:+7.1f}° {status}")

            # 检查/imu/data的yaw
            if self.imu_data:
                imu_yaw = math.degrees(np.mean([d['yaw'] for d in list(self.imu_data)[-100:]]))
                diff = normalize_angle(imu_yaw - gps_heading)
                print(f"    /imu/data : Yaw={imu_yaw:+7.1f}°, 与GPS差={diff:+7.1f}°")

            # 检查LIO-SAM IMU的yaw
            if self.lio_imu_data:
                lio_yaw = math.degrees(np.mean([d['yaw'] for d in list(self.lio_imu_data)[-100:]]))
                diff = normalize_angle(lio_yaw - gps_heading)
                print(f"    LIO-IMU   : Yaw={lio_yaw:+7.1f}°, 与GPS差={diff:+7.1f}°")

        # ============================================
        # 3. 分析正确的外参配置
        # ============================================
        print("\n[3. 外参配置建议]")

        # 确定extrinsicRot (加速度变换)
        print("\n  [extrinsicRot - 加速度/角速度变换]")
        best_rot = None
        for name, ok in gravity_check.items():
            if ok:
                best_rot = name
                break

        if best_rot:
            print(f"    建议使用: {best_rot}")
            if best_rot == 'Identity':
                print("    参数: [1,0,0, 0,1,0, 0,0,1]")
            elif best_rot == 'Rz(180°)':
                print("    参数: [-1,0,0, 0,-1,0, 0,0,1]")
            elif best_rot == 'Rz(90°)':
                print("    参数: [0,-1,0, 1,0,0, 0,0,1]")
            elif best_rot == 'Rz(-90°)':
                print("    ���数: [0,1,0, -1,0,0, 0,0,1]")
        else:
            print("    警告: 没有找到合适的旋转矩阵使重力在-Z方向")
            print("    可能需要自定义旋转矩阵")

        # 确定extrinsicRPY (姿态变换) - 基于yaw差异
        print("\n  [extrinsicRPY - 姿态变换]")

        if len(valid_gps) >= 10 and yaw_results:
            # 找到与GPS航向最接近的积分器
            best_integrator = min(yaw_results.items(), key=lambda x: abs(x[1]['diff']))
            print(f"    最佳积分配置: {best_integrator[0]} (差异 {best_integrator[1]['diff']:.1f}°)")

            # 计算需要的yaw偏移
            # 使用extRot积分的结果作为基准
            if 'extRot' in yaw_results:
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

        # ============================================
        # 4. CORRIMU状态信息
        # ============================================
        if self.corrimu_data:
            latest = self.corrimu_data[-1]
            print(f"\n[4. CORRIMU状态]")
            print(f"  bias_comp: {latest['bias_comp']}")
            print(f"  imu_status: {latest['imu_status']}")

        # ============================================
        # 5. 综合建议
        # ============================================
        print("\n" + "=" * 80)
        print("[综合建议]")
        print("=" * 80)

        print("""
对于 FpaImu (CORRIMU) 数据:

1. CORRIMU 没有原生的四元数数据，需要从角速度积分
2. imuPreintegration.cpp 中的 fpaImuHandler 负责积分
3. 积分使用的是 extRot 变换后的角速度

关键配置:
- extrinsicRot: 应用于加速度和角速度，目的是使重力在-Z方向
- extrinsicRPY: 不直接用于FpaImu(因为没有原生姿态)，
                但角速度积分后的四元数需要与GPS航向一致

注意: 当前代码中，角速度积分使用的是 thisImu (已被extRot变换后的)，
      所以积分出的四元数已经在LiDAR坐标系下。
""")


def main():
    if not HAS_FPAIMU:
        print("ERROR: fixposition_driver_msgs not available!")
        print("Make sure to source the workspace with fixposition_driver_msgs")
        return

    verifier = FpaImuQuaternionVerifier()
    rospy.spin()


if __name__ == '__main__':
    main()
