#!/usr/bin/env python3
"""
验证IMU外参脚本
分别检测 /fixposition/fpa/corrimu, /imu/data 和 /lio_sam/imu/data 的坐标系
通过比较加速度方向和GPS速度方向来判断正确的外参
评估四元数积分姿态的正确性
"""

import rospy
import math
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from collections import deque

from fixposition_driver_msgs.msg import FpaImu


def quaternion_to_euler(qx, qy, qz, qw):
    """四元数转欧拉角 (roll, pitch, yaw)"""
    # Roll (x-axis rotation)
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
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


class IMUExtrinsicsVerifier:
    def __init__(self):
        rospy.init_node('imu_extrinsics_verifier', anonymous=True)

        # 数据缓存
        self.corrimu_data = deque(maxlen=200)
        self.imu_data = deque(maxlen=200)
        self.lio_imu_data = deque(maxlen=200)  # LIO-SAM积分的IMU数据
        self.gps_data = deque(maxlen=200)

        self.last_gps_pos = None
        self.last_gps_time = None

        # 订阅
        rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.corrimu_cb)
        rospy.Subscriber('/imu/data', Imu, self.imu_cb)
        rospy.Subscriber('/lio_sam/imu/data', Imu, self.lio_imu_cb)  # 新增: LIO-SAM积分的IMU
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb)

        # 定时分析
        rospy.Timer(rospy.Duration(3.0), self.analyze)

        print("=" * 70)
        print("IMU外参验证工具 (含四元数积分评估)")
        print("=" * 70)
        print("订阅话题:")
        print("  - /fixposition/fpa/corrimu (CORRIMU原始数据)")
        print("  - /imu/data (标准IMU)")
        print("  - /lio_sam/imu/data (LIO-SAM积分姿态)")
        print("  - /odometry/gps (GPS)")
        print()
        print("分析方法:")
        print("  1. 比较IMU加速度方向和GPS速度变化方向")
        print("  2. 评估四元数积分姿态与GPS航向的一致性")
        print("  3. 检查姿态角变化率与角速度的一致性")
        print("=" * 70)

    def corrimu_cb(self, msg):
        self.corrimu_data.append({
            'time': msg.data.header.stamp.to_sec(),
            'acc_x': msg.data.linear_acceleration.x,
            'acc_y': msg.data.linear_acceleration.y,
            'acc_z': msg.data.linear_acceleration.z,
            'gyr_x': msg.data.angular_velocity.x,
            'gyr_y': msg.data.angular_velocity.y,
            'gyr_z': msg.data.angular_velocity.z,
        })

    def imu_cb(self, msg):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

        self.imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc_x': msg.linear_acceleration.x,
            'acc_y': msg.linear_acceleration.y,
            'acc_z': msg.linear_acceleration.z,
            'gyr_x': msg.angular_velocity.x,
            'gyr_y': msg.angular_velocity.y,
            'gyr_z': msg.angular_velocity.z,
            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
        })

    def lio_imu_cb(self, msg):
        """处理LIO-SAM发布的带积分姿态的IMU数据"""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

        # 检查四元数是否有效
        quat_norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)

        self.lio_imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc_x': msg.linear_acceleration.x,
            'acc_y': msg.linear_acceleration.y,
            'acc_z': msg.linear_acceleration.z,
            'gyr_x': msg.angular_velocity.x,
            'gyr_y': msg.angular_velocity.y,
            'gyr_z': msg.angular_velocity.z,
            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
            'quat_norm': quat_norm,
            'roll': roll, 'pitch': pitch, 'yaw': yaw,
            'orientation_cov': msg.orientation_covariance[0],
        })

    def gps_cb(self, msg):
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 提取四元数
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

        # 计算速度
        vx, vy = 0, 0
        if self.last_gps_pos is not None:
            dt = t - self.last_gps_time
            if dt > 0.01:
                vx = (x - self.last_gps_pos[0]) / dt
                vy = (y - self.last_gps_pos[1]) / dt

        self.last_gps_pos = (x, y)
        self.last_gps_time = t

        self.gps_data.append({
            'time': t,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
        })

    def analyze(self, event):
        print("\n" + "=" * 70)
        print(f"数据统计: CORRIMU={len(self.corrimu_data)}, /imu/data={len(self.imu_data)}, "
              f"LIO-IMU={len(self.lio_imu_data)}, GPS={len(self.gps_data)}")
        print("=" * 70)

        # 1. 分析CORRIMU
        if len(self.corrimu_data) > 10:
            self.analyze_corrimu()

        # 2. 分析/imu/data
        if len(self.imu_data) > 10:
            self.analyze_imu_data()

        # 3. 分析LIO-SAM积分的IMU姿态
        if len(self.lio_imu_data) > 10:
            self.analyze_lio_imu_data()

        # 4. 分析GPS
        if len(self.gps_data) > 10:
            self.analyze_gps()

        # 5. 四元数积分评估
        if len(self.lio_imu_data) > 10 and len(self.gps_data) > 10:
            self.evaluate_quaternion_integration()

        # 6. 外参建议
        self.suggest_extrinsics()

    def analyze_corrimu(self):
        print("\n[CORRIMU 原始数据分析]")
        acc_x = np.mean([d['acc_x'] for d in self.corrimu_data])
        acc_y = np.mean([d['acc_y'] for d in self.corrimu_data])
        acc_z = np.mean([d['acc_z'] for d in self.corrimu_data])
        gyr_x = np.mean([d['gyr_x'] for d in self.corrimu_data])
        gyr_y = np.mean([d['gyr_y'] for d in self.corrimu_data])
        gyr_z = np.mean([d['gyr_z'] for d in self.corrimu_data])

        print(f"  平均加速度: X={acc_x:.3f}, Y={acc_y:.3f}, Z={acc_z:.3f} m/s²")
        print(f"  平均角速度: X={gyr_x:.4f}, Y={gyr_y:.4f}, Z={gyr_z:.4f} rad/s")

        # 重力方向检测
        acc_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        print(f"  加速度模长: {acc_mag:.3f} m/s² (应≈9.8)")

        if abs(acc_z) > 8:
            print(f"  重力方向: Z轴 ({'向下(+Z)' if acc_z > 0 else '向上(-Z)'})")
        elif abs(acc_x) > 8:
            print(f"  重力方向: X轴 (需要检查安装)")
        elif abs(acc_y) > 8:
            print(f"  重力方向: Y轴 (需要检查安装)")

    def analyze_imu_data(self):
        print("\n[/imu/data 分析]")
        acc_x = np.mean([d['acc_x'] for d in self.imu_data])
        acc_y = np.mean([d['acc_y'] for d in self.imu_data])
        acc_z = np.mean([d['acc_z'] for d in self.imu_data])
        roll_mean = np.mean([d['roll'] for d in self.imu_data])
        pitch_mean = np.mean([d['pitch'] for d in self.imu_data])
        yaw_mean = np.mean([d['yaw'] for d in self.imu_data])

        print(f"  平均加速度: X={acc_x:.3f}, Y={acc_y:.3f}, Z={acc_z:.3f} m/s²")
        print(f"  平均姿态: Roll={math.degrees(roll_mean):.1f}°, "
              f"Pitch={math.degrees(pitch_mean):.1f}°, Yaw={math.degrees(yaw_mean):.1f}°")

        acc_mag = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        print(f"  加速度模长: {acc_mag:.3f} m/s² (应≈9.8)")

    def analyze_lio_imu_data(self):
        print("\n[LIO-SAM积分IMU姿态分析] (话题: /lio_sam/imu/data)")

        # 基本统计
        acc_x = np.mean([d['acc_x'] for d in self.lio_imu_data])
        acc_y = np.mean([d['acc_y'] for d in self.lio_imu_data])
        acc_z = np.mean([d['acc_z'] for d in self.lio_imu_data])

        roll_mean = np.mean([d['roll'] for d in self.lio_imu_data])
        pitch_mean = np.mean([d['pitch'] for d in self.lio_imu_data])
        yaw_mean = np.mean([d['yaw'] for d in self.lio_imu_data])

        roll_std = np.std([d['roll'] for d in self.lio_imu_data])
        pitch_std = np.std([d['pitch'] for d in self.lio_imu_data])
        yaw_std = np.std([d['yaw'] for d in self.lio_imu_data])

        # 四元数范数检查
        quat_norms = [d['quat_norm'] for d in self.lio_imu_data]
        quat_norm_mean = np.mean(quat_norms)
        quat_norm_std = np.std(quat_norms)

        print(f"  平均加速度: X={acc_x:.3f}, Y={acc_y:.3f}, Z={acc_z:.3f} m/s²")
        print(f"  平均姿态: Roll={math.degrees(roll_mean):.2f}°, "
              f"Pitch={math.degrees(pitch_mean):.2f}°, Yaw={math.degrees(yaw_mean):.2f}°")
        print(f"  姿态标准差: Roll={math.degrees(roll_std):.2f}°, "
              f"Pitch={math.degrees(pitch_std):.2f}°, Yaw={math.degrees(yaw_std):.2f}°")

        # 四元数验证
        print(f"\n  四元数范数: 均值={quat_norm_mean:.6f}, 标准差={quat_norm_std:.6f}")
        if abs(quat_norm_mean - 1.0) < 0.001 and quat_norm_std < 0.001:
            print("  ✓ 四元数归一化正确")
        else:
            print("  ✗ 四元数归一化异常!")

        # 检查姿态变化率与角速度一致性
        if len(self.lio_imu_data) > 20:
            self.check_integration_consistency()

    def check_integration_consistency(self):
        """检查姿态积分与角速度的一致性"""
        print("\n  [积分一致性检查]")

        data = list(self.lio_imu_data)

        # 收集所有轴的数据
        roll_rates_diff = []
        pitch_rates_diff = []
        yaw_rates_diff = []
        gyr_x_list = []
        gyr_y_list = []
        gyr_z_list = []

        for i in range(1, len(data)):
            dt = data[i]['time'] - data[i-1]['time']
            if dt > 0.001 and dt < 0.1:
                # Roll变化率
                droll = data[i]['roll'] - data[i-1]['roll']
                if droll > math.pi: droll -= 2 * math.pi
                elif droll < -math.pi: droll += 2 * math.pi
                roll_rates_diff.append(droll / dt)

                # Pitch变化率
                dpitch = data[i]['pitch'] - data[i-1]['pitch']
                if dpitch > math.pi: dpitch -= 2 * math.pi
                elif dpitch < -math.pi: dpitch += 2 * math.pi
                pitch_rates_diff.append(dpitch / dt)

                # Yaw变化率
                dyaw = data[i]['yaw'] - data[i-1]['yaw']
                if dyaw > math.pi: dyaw -= 2 * math.pi
                elif dyaw < -math.pi: dyaw += 2 * math.pi
                yaw_rates_diff.append(dyaw / dt)

                gyr_x_list.append(data[i]['gyr_x'])
                gyr_y_list.append(data[i]['gyr_y'])
                gyr_z_list.append(data[i]['gyr_z'])

        if yaw_rates_diff and len(yaw_rates_diff) > 10:
            # 计算各轴均值
            print(f"    姿态差分均值: dRoll={np.mean(roll_rates_diff):.4f}, "
                  f"dPitch={np.mean(pitch_rates_diff):.4f}, dYaw={np.mean(yaw_rates_diff):.4f} rad/s")
            print(f"    陀螺仪均值:   gyr_x={np.mean(gyr_x_list):.4f}, "
                  f"gyr_y={np.mean(gyr_y_list):.4f}, gyr_z={np.mean(gyr_z_list):.4f} rad/s")

            # 计算相关系数矩阵 (检查是否有轴错位)
            print("\n    相关系数矩阵 (姿态差分 vs 陀螺仪):")
            print("              gyr_x    gyr_y    gyr_z")

            corr_roll_x = np.corrcoef(roll_rates_diff, gyr_x_list)[0, 1] if np.std(roll_rates_diff) > 0.001 else 0
            corr_roll_y = np.corrcoef(roll_rates_diff, gyr_y_list)[0, 1] if np.std(roll_rates_diff) > 0.001 else 0
            corr_roll_z = np.corrcoef(roll_rates_diff, gyr_z_list)[0, 1] if np.std(roll_rates_diff) > 0.001 else 0
            print(f"    dRoll:    {corr_roll_x:7.3f}  {corr_roll_y:7.3f}  {corr_roll_z:7.3f}")

            corr_pitch_x = np.corrcoef(pitch_rates_diff, gyr_x_list)[0, 1] if np.std(pitch_rates_diff) > 0.001 else 0
            corr_pitch_y = np.corrcoef(pitch_rates_diff, gyr_y_list)[0, 1] if np.std(pitch_rates_diff) > 0.001 else 0
            corr_pitch_z = np.corrcoef(pitch_rates_diff, gyr_z_list)[0, 1] if np.std(pitch_rates_diff) > 0.001 else 0
            print(f"    dPitch:   {corr_pitch_x:7.3f}  {corr_pitch_y:7.3f}  {corr_pitch_z:7.3f}")

            corr_yaw_x = np.corrcoef(yaw_rates_diff, gyr_x_list)[0, 1] if np.std(yaw_rates_diff) > 0.001 else 0
            corr_yaw_y = np.corrcoef(yaw_rates_diff, gyr_y_list)[0, 1] if np.std(yaw_rates_diff) > 0.001 else 0
            corr_yaw_z = np.corrcoef(yaw_rates_diff, gyr_z_list)[0, 1] if np.std(yaw_rates_diff) > 0.001 else 0
            print(f"    dYaw:     {corr_yaw_x:7.3f}  {corr_yaw_y:7.3f}  {corr_yaw_z:7.3f}")

            # 理想情况: dRoll~gyr_x, dPitch~gyr_y, dYaw~gyr_z 对角线应该接近1
            diag_corr = [corr_roll_x, corr_pitch_y, corr_yaw_z]
            print(f"\n    对角线相关性: [{corr_roll_x:.3f}, {corr_pitch_y:.3f}, {corr_yaw_z:.3f}]")

            if all(c > 0.8 for c in diag_corr):
                print("    ✓ 积分一致性良好，坐标轴对齐正确")
            elif corr_yaw_z > 0.8:
                print("    △ Yaw积分正确，Roll/Pitch可能有问题")
            else:
                # 检查是否存在轴错位
                max_corr_yaw = max(abs(corr_yaw_x), abs(corr_yaw_y), abs(corr_yaw_z))
                if abs(corr_yaw_x) == max_corr_yaw:
                    print("    ✗ Yaw与gyr_x相关性最高，可能存在轴错位!")
                elif abs(corr_yaw_y) == max_corr_yaw:
                    print("    ✗ Yaw与gyr_y相关性最高，可能存在轴错位!")
                else:
                    print("    ✗ 积分一致性差，检查四元数积分实现!")

    def analyze_gps(self):
        print("\n[GPS 分析]")
        # 取最近的有效速度数据
        valid_gps = [d for d in self.gps_data if math.sqrt(d['vx']**2 + d['vy']**2) > 0.5]

        if valid_gps:
            vx_mean = np.mean([d['vx'] for d in valid_gps])
            vy_mean = np.mean([d['vy'] for d in valid_gps])
            yaw_mean = np.mean([d['yaw'] for d in valid_gps])
            roll_mean = np.mean([d['roll'] for d in valid_gps])
            pitch_mean = np.mean([d['pitch'] for d in valid_gps])

            speed = math.sqrt(vx_mean**2 + vy_mean**2)
            vel_heading = math.degrees(math.atan2(vy_mean, vx_mean))

            print(f"  平均速度: Vx={vx_mean:.2f}, Vy={vy_mean:.2f} m/s (速度={speed:.2f} m/s)")
            print(f"  速度航向: {vel_heading:.1f}°")
            print(f"  GPS姿态: Roll={math.degrees(roll_mean):.1f}°, "
                  f"Pitch={math.degrees(pitch_mean):.1f}°, Yaw={math.degrees(yaw_mean):.1f}°")
            print(f"  航向差(姿态-速度): {normalize_angle(math.degrees(yaw_mean) - vel_heading):.1f}°")
        else:
            print("  速度数据不足 (需要 > 0.5 m/s)")

    def evaluate_quaternion_integration(self):
        """评估四元数积分姿态的正确性"""
        print("\n" + "=" * 70)
        print("[四元数积分评估]")
        print("=" * 70)

        lio_data = list(self.lio_imu_data)
        gps_data = list(self.gps_data)

        if not lio_data or not gps_data:
            print("  数据不足，无法评估")
            return

        # 1. 比较最近时刻的姿态
        lio_latest = lio_data[-1]
        gps_valid = [d for d in gps_data if math.sqrt(d['vx']**2 + d['vy']**2) > 0.3]

        print(f"\n  [最新积分姿态]")
        print(f"    四元数: ({lio_latest['qw']:.4f}, {lio_latest['qx']:.4f}, "
              f"{lio_latest['qy']:.4f}, {lio_latest['qz']:.4f})")
        print(f"    欧拉角: Roll={math.degrees(lio_latest['roll']):.2f}°, "
              f"Pitch={math.degrees(lio_latest['pitch']):.2f}°, "
              f"Yaw={math.degrees(lio_latest['yaw']):.2f}°")

        if gps_valid:
            gps_latest = gps_valid[-1]
            lio_yaw_deg = math.degrees(lio_latest['yaw'])
            gps_yaw_deg = math.degrees(gps_latest['yaw'])
            yaw_diff = normalize_angle(lio_yaw_deg - gps_yaw_deg)

            print(f"\n  [与GPS航向比较]")
            print(f"    LIO-SAM积分Yaw: {lio_yaw_deg:.2f}°")
            print(f"    GPS Yaw: {gps_yaw_deg:.2f}°")
            print(f"    差值: {yaw_diff:.2f}°")

            if abs(yaw_diff) < 10:
                print("    ✓ 航向一致性良好")
            elif abs(yaw_diff) < 30:
                print("    △ 航向存在偏差，可能是初始化或漂移")
            else:
                print("    ✗ 航向差异较大，检查外参或积分实现")

        # 2. 检查Roll/Pitch是否合理（静态时应接近0）
        print(f"\n  [Roll/Pitch合理性]")
        roll_deg = math.degrees(lio_latest['roll'])
        pitch_deg = math.degrees(lio_latest['pitch'])

        if abs(roll_deg) < 5 and abs(pitch_deg) < 5:
            print(f"    ✓ Roll/Pitch在合理范围内 (Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°)")
        elif abs(roll_deg) < 15 and abs(pitch_deg) < 15:
            print(f"    △ Roll/Pitch略有偏差 (Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°)")
        else:
            print(f"    ✗ Roll/Pitch异常 (Roll={roll_deg:.2f}°, Pitch={pitch_deg:.2f}°)")
            print("       可能原因: 外参设置错误或重力方向判断错误")

        # 3. 计算Yaw漂移率
        if len(lio_data) > 50:
            time_span = lio_data[-1]['time'] - lio_data[0]['time']
            if time_span > 1.0:
                yaw_start = lio_data[0]['yaw']
                yaw_end = lio_data[-1]['yaw']
                yaw_change = yaw_end - yaw_start
                # 处理角度跳变
                if yaw_change > math.pi:
                    yaw_change -= 2 * math.pi
                elif yaw_change < -math.pi:
                    yaw_change += 2 * math.pi

                drift_rate = math.degrees(yaw_change) / time_span

                print(f"\n  [Yaw漂移率]")
                print(f"    时间跨度: {time_span:.1f}s")
                print(f"    Yaw变化: {math.degrees(yaw_change):.2f}°")
                print(f"    漂移率: {drift_rate:.3f} °/s")

                if abs(drift_rate) < 0.5:
                    print("    ✓ 漂移率在可接受范围内")
                else:
                    print("    △ 漂移率较高，积分误差可能累积")

    def suggest_extrinsics(self):
        print("\n" + "=" * 70)
        print("[外参建议]")
        print("=" * 70)

        # 使用LIO-SAM积分的IMU数据进行分析
        if len(self.lio_imu_data) > 10 and len(self.gps_data) > 10:
            lio_yaw = math.degrees(np.mean([d['yaw'] for d in self.lio_imu_data]))
            gps_yaws = [d['yaw'] for d in self.gps_data if abs(d['yaw']) > 0.01]

            if gps_yaws:
                gps_yaw = math.degrees(np.mean(gps_yaws))
                yaw_diff = normalize_angle(lio_yaw - gps_yaw)

                print(f"\n  LIO-SAM积分Yaw: {lio_yaw:.1f}°")
                print(f"  GPS Yaw: {gps_yaw:.1f}°")
                print(f"  差值: {yaw_diff:.1f}°")

                if abs(yaw_diff) < 15:
                    print("\n  ✓ 外参设置正确，航向一致")
                elif abs(yaw_diff - 90) < 15 or abs(yaw_diff + 270) < 15:
                    print("\n  建议: 需要增加90°的Yaw偏移")
                elif abs(yaw_diff + 90) < 15 or abs(yaw_diff - 270) < 15:
                    print("\n  建议: 需要减少90°的Yaw偏移")
                elif abs(abs(yaw_diff) - 180) < 15:
                    print("\n  建议: 需要增加180°的Yaw偏移")
                else:
                    print(f"\n  △ 航向差异 {yaw_diff:.1f}° 不是典型值")
                    print("     请检查传感器安装或外参矩阵")

        print("\n  注意事项:")
        print("  - CORRIMU没有姿态数据，依赖角速度积分")
        print("  - 积分姿态会漂移，需要通过LiDAR里程计或GPS校正")
        print("  - 如果漂移严重，检查陀螺仪偏置和外参")


def main():
    verifier = IMUExtrinsicsVerifier()
    rospy.spin()



if __name__ == '__main__':
    main()
