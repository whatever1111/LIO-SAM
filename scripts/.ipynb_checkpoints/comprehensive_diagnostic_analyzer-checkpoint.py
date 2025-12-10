#!/usr/bin/env python3
"""
LIO-SAM Comprehensive Diagnostic Analyzer
=========================================
深入分析LIO-SAM运行诊断数据,包括:
1. 协方差分析 (位置、姿态、速度、偏置)
2. GPS状态分析 (左/右轨迹、融合效果)
3. IMU预积分分析
4. 因子图健康度分析
5. 优化建议生成

基于LIO-SAM原理设计:
- ISAM2增量优化: poseCovariance来自marginalCovariance
- GPS Factor: 当poseCovariance > poseCovThreshold时添加
- IMU Preintegration: 使用GTSAM的PreintegratedImuMeasurements

Author: Claude
Date: 2024
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from collections import defaultdict
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class LIOSAMDiagnosticAnalyzer:
    """LIO-SAM诊断数据综合分析器"""

    def __init__(self, diagnostic_csv=None, output_dir=None):
        self.diagnostic_csv = diagnostic_csv
        self.output_dir = output_dir or '/root/autodl-tmp/catkin_ws/src/LIO-SAM/output'

        # 数据容器
        self.diagnostic_data = None
        self.gps_trajectory = None
        self.fusion_trajectory = None
        self.imu_orientation = None
        self.imu_acceleration = None
        self.imu_angular_velocity = None
        self.gps_latency = None
        self.fusion_latency = None

        # 分析结果
        self.analysis_results = {}
        self.optimization_suggestions = []

    def load_all_data(self):
        """加载所有可用数据"""
        print("="*60)
        print("Loading diagnostic data...")
        print("="*60)

        # 加载诊断CSV
        if self.diagnostic_csv and os.path.exists(self.diagnostic_csv):
            self.diagnostic_data = pd.read_csv(self.diagnostic_csv)
            print(f"Loaded diagnostic data: {len(self.diagnostic_data)} records")

        # 加载轨迹数据
        data_files = {
            'gps_trajectory': 'gps_trajectory.csv',
            'fusion_trajectory': 'fusion_trajectory.csv',
            'imu_orientation': 'imu_orientation.csv',
            'imu_acceleration': 'imu_linear_acceleration.csv',
            'imu_angular_velocity': 'imu_angular_velocity.csv',
            'gps_latency': 'gps_latency.csv',
            'fusion_latency': 'fusion_latency.csv',
        }

        for attr, filename in data_files.items():
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                try:
                    data = pd.read_csv(filepath)
                    setattr(self, attr, data)
                    print(f"Loaded {filename}: {len(data)} records")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

        print()

    def analyze_diagnostic_events(self):
        """分析诊断事件"""
        if self.diagnostic_data is None or len(self.diagnostic_data) == 0:
            print("No diagnostic data available")
            return

        print("="*60)
        print("Analyzing Diagnostic Events")
        print("="*60)

        # 按事件类型分组统计
        event_counts = self.diagnostic_data.groupby(['source', 'type']).size()
        print("\nEvent Summary:")
        print(event_counts.to_string())

        # 分析异常值统计
        anomaly_stats = {}
        for (source, event_type), group in self.diagnostic_data.groupby(['source', 'type']):
            values = group['value'].values
            anomaly_stats[f"{source}_{event_type}"] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        self.analysis_results['anomaly_stats'] = anomaly_stats

        # 分析时间分布
        if 'timestamp' in self.diagnostic_data.columns:
            timestamps = self.diagnostic_data['timestamp'].values
            time_range = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            print(f"\nTime range: {time_range:.2f} seconds")

            # 计算事件频率
            if time_range > 0:
                event_rate = len(timestamps) / time_range
                print(f"Average event rate: {event_rate:.2f} events/second")

        print()

    def analyze_covariance_evolution(self):
        """
        分析协方差演化

        LIO-SAM中的协方差来源:
        1. poseCovariance: ISAM2.marginalCovariance(key) 返回的6x6矩阵
           - [0:3, 0:3]: 旋转协方差 (roll, pitch, yaw)
           - [3:6, 3:6]: 平移协方差 (x, y, z)
        2. GPS协方差: 来自/odometry/gps消息的covariance字段
        3. IMU预积分协方差: PreintegratedImuMeasurements内部维护
        """
        print("="*60)
        print("Analyzing Covariance Evolution")
        print("="*60)

        # 从诊断数据提取协方差相关信息
        if self.diagnostic_data is not None:
            # 提取位置跳变数据（间接反映协方差问题）
            pos_jumps = self.diagnostic_data[
                self.diagnostic_data['type'] == 'POS_JUMP'
            ]['value'].values

            if len(pos_jumps) > 0:
                print(f"\nPosition Jump Analysis:")
                print(f"  Count: {len(pos_jumps)}")
                print(f"  Mean implied velocity: {np.mean(pos_jumps):.2f} m/s")
                print(f"  Max implied velocity: {np.max(pos_jumps):.2f} m/s")

                # 判断是否存在协方差发散问题
                if np.max(pos_jumps) > 100:
                    self.optimization_suggestions.append(
                        "CRITICAL: Position jumps > 100 m/s detected. "
                        "This indicates covariance divergence or sensor failures."
                    )

            # 提取速度异常数据
            vel_anomalies = self.diagnostic_data[
                self.diagnostic_data['type'] == 'VEL_ANOMALY'
            ]['value'].values

            if len(vel_anomalies) > 0:
                print(f"\nVelocity Anomaly Analysis:")
                print(f"  Count: {len(vel_anomalies)}")
                print(f"  Mean: {np.mean(vel_anomalies):.2f} m/s")
                print(f"  Max: {np.max(vel_anomalies):.2f} m/s")
                print(f"  Growth rate: {(vel_anomalies[-1] - vel_anomalies[0]) / len(vel_anomalies):.4f} m/s per event")

        # 基于GPS和融合轨迹分析位置协方差效果
        if self.gps_trajectory is not None and self.fusion_trajectory is not None:
            self._analyze_trajectory_covariance()

        print()

    def _analyze_trajectory_covariance(self):
        """通过GPS和融合轨迹分析协方差效果 - 使用时间戳插值对齐"""
        gps = self.gps_trajectory
        fusion = self.fusion_trajectory

        # 确保有足够的数据
        if len(gps) < 10 or len(fusion) < 10:
            return

        gps_times = gps['time'].values
        fusion_times = fusion['time'].values

        # 找到重叠的时间范围
        t_start = max(gps_times[0], fusion_times[0])
        t_end = min(gps_times[-1], fusion_times[-1])

        if t_end <= t_start:
            print("  Warning: No overlapping time range between GPS and Fusion")
            return

        # 创建GPS插值函数
        try:
            gps_interp_x = interpolate.interp1d(gps_times, gps['x'].values,
                                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            gps_interp_y = interpolate.interp1d(gps_times, gps['y'].values,
                                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            gps_interp_z = interpolate.interp1d(gps_times, gps['z'].values,
                                                 kind='linear', bounds_error=False, fill_value='extrapolate')
        except Exception as e:
            print(f"  Warning: Failed to create interpolation: {e}")
            return

        # 使用fusion的时间戳作为基准，筛选重叠时间范围内的点
        valid_mask = (fusion_times >= t_start) & (fusion_times <= t_end)
        valid_fusion_times = fusion_times[valid_mask]

        if len(valid_fusion_times) < 10:
            print("  Warning: Not enough overlapping points")
            return

        # 在fusion时间点插值GPS位置
        gps_x_aligned = gps_interp_x(valid_fusion_times)
        gps_y_aligned = gps_interp_y(valid_fusion_times)
        gps_z_aligned = gps_interp_z(valid_fusion_times)

        # 计算对齐后的位置误差
        fusion_x = fusion['x'].values[valid_mask]
        fusion_y = fusion['y'].values[valid_mask]
        fusion_z = fusion['z'].values[valid_mask]

        dx = fusion_x - gps_x_aligned
        dy = fusion_y - gps_y_aligned
        dz = fusion_z - gps_z_aligned

        position_error = np.sqrt(dx**2 + dy**2 + dz**2)
        position_error_2d = np.sqrt(dx**2 + dy**2)  # 2D误差更有意义

        # 保存对齐后的数据用于绘图
        self.analysis_results['aligned_trajectory'] = {
            'times': valid_fusion_times,
            'fusion_x': fusion_x,
            'fusion_y': fusion_y,
            'fusion_z': fusion_z,
            'gps_x': gps_x_aligned,
            'gps_y': gps_y_aligned,
            'gps_z': gps_z_aligned,
            'dx': dx,
            'dy': dy,
            'dz': dz
        }

        print(f"\nGPS-Fusion Position Error (Time-Aligned):")
        print(f"  Aligned points: {len(position_error)}")
        print(f"  3D Error - Mean: {np.mean(position_error):.3f} m, Std: {np.std(position_error):.3f} m, Max: {np.max(position_error):.3f} m")
        print(f"  2D Error - Mean: {np.mean(position_error_2d):.3f} m, Std: {np.std(position_error_2d):.3f} m, Max: {np.max(position_error_2d):.3f} m")

        # 计算终点误差
        end_error = np.sqrt((fusion_x[-1] - gps_x_aligned[-1])**2 +
                           (fusion_y[-1] - gps_y_aligned[-1])**2 +
                           (fusion_z[-1] - gps_z_aligned[-1])**2)
        end_error_2d = np.sqrt((fusion_x[-1] - gps_x_aligned[-1])**2 +
                               (fusion_y[-1] - gps_y_aligned[-1])**2)
        print(f"  End Point Error - 3D: {end_error:.3f} m, 2D: {end_error_2d:.3f} m")

        self.analysis_results['gps_fusion_error'] = {
            'mean': np.mean(position_error),
            'mean_2d': np.mean(position_error_2d),
            'std': np.std(position_error),
            'std_2d': np.std(position_error_2d),
            'max': np.max(position_error),
            'max_2d': np.max(position_error_2d),
            'end_error': end_error,
            'end_error_2d': end_error_2d
        }

        # 判断GPS融合效果 - 使用更合理的阈值
        if np.mean(position_error_2d) > 2.0:
            self.optimization_suggestions.append(
                f"GPS-Fusion mean 2D error ({np.mean(position_error_2d):.2f}m) is relatively high. "
                "Consider adjusting gpsCovThreshold or poseCovThreshold."
            )
        elif np.mean(position_error_2d) < 0.5:
            print("  Status: EXCELLENT - GPS and Fusion are well aligned")

    def analyze_gps_status(self):
        """
        分析GPS状态

        LIO-SAM中GPS Factor添加条件:
        1. gpsQueue非空
        2. 已有keyframe
        3. travel_dist >= 5m (或gpsAddInterval设置)
        4. poseCovariance(3,3) >= poseCovThreshold OR poseCovariance(4,4) >= poseCovThreshold
        5. GPS covariance < gpsCovThreshold
        6. GPS位置有效 (非0,0,0)
        7. 距离上次GPS > gpsAddInterval
        """
        print("="*60)
        print("Analyzing GPS Status")
        print("="*60)

        if self.gps_trajectory is None:
            print("No GPS trajectory data")
            return

        gps = self.gps_trajectory

        # 基本统计
        print(f"\nGPS Trajectory Statistics:")
        print(f"  Total points: {len(gps)}")
        print(f"  X range: [{gps['x'].min():.2f}, {gps['x'].max():.2f}] m")
        print(f"  Y range: [{gps['y'].min():.2f}, {gps['y'].max():.2f}] m")
        print(f"  Z range: [{gps['z'].min():.2f}, {gps['z'].max():.2f}] m")

        # 计算GPS轨迹长度
        if len(gps) > 1:
            dx = np.diff(gps['x'].values)
            dy = np.diff(gps['y'].values)
            dz = np.diff(gps['z'].values)
            segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
            total_distance = np.sum(segment_lengths)
            print(f"  Total distance: {total_distance:.2f} m")

            # 检测GPS跳变
            jump_threshold = 5.0  # meters per sample
            jumps = segment_lengths > jump_threshold
            if np.any(jumps):
                print(f"  GPS jumps detected: {np.sum(jumps)} times")
                print(f"  Max jump: {np.max(segment_lengths):.2f} m")
                self.optimization_suggestions.append(
                    f"GPS has {np.sum(jumps)} position jumps. "
                    "Consider increasing gpsCovThreshold to filter noisy GPS."
                )

        # GPS延迟分析
        if self.gps_latency is not None:
            latency = self.gps_latency['latency_ms'].values
            print(f"\nGPS Latency:")
            print(f"  Mean: {np.mean(latency):.2f} ms")
            print(f"  Std: {np.std(latency):.2f} ms")
            print(f"  Max: {np.max(latency):.2f} ms")

            if np.mean(latency) > 100:
                self.optimization_suggestions.append(
                    f"GPS latency ({np.mean(latency):.0f}ms) is high. "
                    "This may cause synchronization issues with LiDAR."
                )

        print()

    def analyze_imu_preintegration(self):
        """
        分析IMU预积分

        LIO-SAM中IMU预积分:
        1. imuIntegratorOpt_: 用于优化的预积分器
        2. imuIntegratorImu_: 用于IMU里程计发布的预积分器
        3. 预积分使用GTSAM PreintegratedImuMeasurements
        4. 参数: imuAccNoise, imuGyrNoise, imuAccBiasN, imuGyrBiasN
        """
        print("="*60)
        print("Analyzing IMU Preintegration")
        print("="*60)

        # 分析IMU数据
        if self.imu_acceleration is not None:
            acc = self.imu_acceleration
            print(f"\nIMU Linear Acceleration:")
            print(f"  Samples: {len(acc)}")

            ax = acc['ax_m_s2'].values
            ay = acc['ay_m_s2'].values
            az = acc['az_m_s2'].values
            acc_norm = np.sqrt(ax**2 + ay**2 + az**2)

            print(f"  X: mean={np.mean(ax):.3f}, std={np.std(ax):.3f} m/s^2")
            print(f"  Y: mean={np.mean(ay):.3f}, std={np.std(ay):.3f} m/s^2")
            print(f"  Z: mean={np.mean(az):.3f}, std={np.std(az):.3f} m/s^2")
            print(f"  Norm: mean={np.mean(acc_norm):.3f}, std={np.std(acc_norm):.3f} m/s^2")

            # 重力估计
            gravity_estimate = np.mean(acc_norm)
            print(f"  Estimated gravity: {gravity_estimate:.3f} m/s^2")

            if abs(gravity_estimate - 9.81) > 0.5:
                self.optimization_suggestions.append(
                    f"IMU gravity estimate ({gravity_estimate:.3f}) differs from 9.81. "
                    "Check imuGravity parameter setting."
                )

            # 检测加速度偏置
            acc_bias_x = np.mean(ax)
            acc_bias_y = np.mean(ay)
            acc_bias_z = np.mean(az) - 9.81  # 假设Z轴向上
            print(f"  Estimated acc bias: [{acc_bias_x:.4f}, {acc_bias_y:.4f}, {acc_bias_z:.4f}] m/s^2")

        if self.imu_angular_velocity is not None:
            gyro = self.imu_angular_velocity
            print(f"\nIMU Angular Velocity:")
            print(f"  Samples: {len(gyro)}")

            wx = gyro['wx_rad_s'].values
            wy = gyro['wy_rad_s'].values
            wz = gyro['wz_rad_s'].values

            print(f"  X: mean={np.mean(wx):.4f}, std={np.std(wx):.4f} rad/s")
            print(f"  Y: mean={np.mean(wy):.4f}, std={np.std(wy):.4f} rad/s")
            print(f"  Z: mean={np.mean(wz):.4f}, std={np.std(wz):.4f} rad/s")

            # 陀螺仪偏置估计
            gyro_bias = np.array([np.mean(wx), np.mean(wy), np.mean(wz)])
            print(f"  Estimated gyro bias: {np.linalg.norm(gyro_bias)*180/np.pi:.4f} deg/s")

        if self.imu_orientation is not None:
            ori = self.imu_orientation
            print(f"\nIMU Orientation:")
            print(f"  Samples: {len(ori)}")

            roll = ori['roll_deg'].values
            pitch = ori['pitch_deg'].values
            yaw = ori['yaw_deg'].values

            print(f"  Roll: [{np.min(roll):.2f}, {np.max(roll):.2f}] deg, std={np.std(roll):.2f}")
            print(f"  Pitch: [{np.min(pitch):.2f}, {np.max(pitch):.2f}] deg, std={np.std(pitch):.2f}")
            print(f"  Yaw: [{np.min(yaw):.2f}, {np.max(yaw):.2f}] deg, std={np.std(yaw):.2f}")

            # 检测yaw漂移 - 通过分析yaw的高频噪声和与GPS航向的一致性
            # 注意：yaw[-1] - yaw[0] 不是漂移，是车辆实际转向
            if len(yaw) > 100:
                time_arr = ori['relative_time_s'].values
                time_range = time_arr[-1] - time_arr[0]

                # 计算yaw变化总量（不是漂移，是实际转向）
                yaw_change = yaw[-1] - yaw[0]
                print(f"  Total yaw change: {yaw_change:.2f} deg over {time_range:.1f}s")

                # 真正的漂移检测：分析短时间窗口内yaw的抖动/噪声
                # 使用滑动窗口计算局部标准差
                window_size = min(50, len(yaw) // 10)
                if window_size > 5:
                    local_stds = []
                    for i in range(0, len(yaw) - window_size, window_size):
                        window = yaw[i:i+window_size]
                        # 去除趋势后的标准差（真正的噪声）
                        detrended = window - np.linspace(window[0], window[-1], len(window))
                        local_stds.append(np.std(detrended))

                    if len(local_stds) > 0:
                        avg_noise = np.mean(local_stds)
                        print(f"  Yaw noise (detrended std): {avg_noise:.4f} deg")

                        if avg_noise > 0.5:
                            self.optimization_suggestions.append(
                                f"IMU yaw noise ({avg_noise:.2f} deg) is high. "
                                "Consider checking gyroscope calibration or increasing imuGyrNoise."
                            )

        # 从诊断数据分析预积分异常
        if self.diagnostic_data is not None:
            preint_anomalies = self.diagnostic_data[
                self.diagnostic_data['source'] == 'IMU_PREINT'
            ]
            if len(preint_anomalies) > 0:
                print(f"\nIMU Preintegration Anomalies:")
                print(f"  Total: {len(preint_anomalies)}")

                vel_values = preint_anomalies['value'].values
                print(f"  Velocity anomaly range: [{np.min(vel_values):.2f}, {np.max(vel_values):.2f}] m/s")

                # 检测速度发散
                if len(vel_values) > 10:
                    velocity_growth = np.polyfit(range(len(vel_values)), vel_values, 1)[0]
                    if velocity_growth > 0.01:
                        self.optimization_suggestions.append(
                            "IMU preintegration velocity is diverging. "
                            "This may indicate IMU noise parameters are too low or bias estimation issues."
                        )

        print()

    def analyze_factor_graph_health(self):
        """
        分析因子图健康度

        LIO-SAM因子图结构:
        1. PriorFactor<Pose3>: 初始位姿先验
        2. PriorFactor<Vector3>: 初始速度先验
        3. PriorFactor<imuBias::ConstantBias>: 初始偏置先验
        4. BetweenFactor<Pose3>: 里程计约束 (来自scan-to-map)
        5. GPSFactor: GPS位置约束
        6. LoopFactor: 回环约束
        7. ImuFactor: IMU预积分约束 (在imuPreintegration中)
        """
        print("="*60)
        print("Analyzing Factor Graph Health")
        print("="*60)

        health_score = 100.0
        issues = []

        # 检查GPS因子效果 - 使用2D误差评估
        if self.gps_trajectory is not None and self.fusion_trajectory is not None:
            error = self.analysis_results.get('gps_fusion_error', {})
            mean_2d = error.get('mean_2d', 0)
            if mean_2d > 5.0:
                health_score -= 20
                issues.append(f"High GPS-Fusion 2D error: {mean_2d:.2f}m")
            elif mean_2d > 2.0:
                health_score -= 10
                issues.append(f"Moderate GPS-Fusion 2D error: {mean_2d:.2f}m")
        else:
            health_score -= 10
            issues.append("Missing trajectory data for GPS factor analysis")

        # 检查位置跳变
        if self.diagnostic_data is not None:
            pos_jumps = self.diagnostic_data[
                self.diagnostic_data['type'] == 'POS_JUMP'
            ]
            if len(pos_jumps) > 10:
                health_score -= min(30, len(pos_jumps))
                issues.append(f"{len(pos_jumps)} position jumps detected")

        # 检查速度异常
        if self.diagnostic_data is not None:
            vel_anomalies = self.diagnostic_data[
                self.diagnostic_data['type'] == 'VEL_ANOMALY'
            ]
            if len(vel_anomalies) > 50:
                health_score -= min(30, len(vel_anomalies) // 10)
                issues.append(f"{len(vel_anomalies)} velocity anomalies detected")

        # 确保分数在0-100范围内
        health_score = max(0, min(100, health_score))

        print(f"\nFactor Graph Health Score: {health_score:.0f}/100")

        if health_score >= 80:
            print("Status: GOOD")
        elif health_score >= 50:
            print("Status: WARNING")
        else:
            print("Status: CRITICAL")

        if issues:
            print("\nIssues detected:")
            for issue in issues:
                print(f"  - {issue}")

        self.analysis_results['health_score'] = health_score
        self.analysis_results['health_issues'] = issues

        print()

    def generate_optimization_suggestions(self):
        """生成优化建议"""
        print("="*60)
        print("Optimization Suggestions")
        print("="*60)

        # 添加基于分析的建议
        health_score = self.analysis_results.get('health_score', 100)

        if health_score < 50:
            self.optimization_suggestions.append(
                "CRITICAL: System health is low. Consider a complete parameter review."
            )

        # GPS相关建议 - 使用2D误差
        gps_error = self.analysis_results.get('gps_fusion_error', {})
        if gps_error.get('mean_2d', 0) > 3.0:
            self.optimization_suggestions.append(
                "Consider reducing gpsNoiseScale to increase GPS weight, "
                "or check GPS extrinsics configuration."
            )

        # 打印所有建议
        if self.optimization_suggestions:
            print("\nBased on the analysis, here are the optimization suggestions:\n")
            for i, suggestion in enumerate(self.optimization_suggestions, 1):
                print(f"{i}. {suggestion}\n")
        else:
            print("\nNo critical issues detected. System parameters appear to be well-configured.")

        # 通用调参建议
        print("\n" + "-"*40)
        print("General Parameter Tuning Guide:")
        print("-"*40)
        print("""
1. GPS Weight Control (params.yaml):
   - gpsCovThreshold: Increase to accept more GPS (current use)
   - poseCovThreshold: Increase to add GPS more frequently
   - gpsNoiseMin: Decrease for higher GPS weight
   - gpsNoiseScale: Decrease for higher GPS weight
   - gpsAddInterval: Decrease for more frequent GPS factors

2. IMU Preintegration (params.yaml):
   - imuAccNoise: Increase if position drifts too fast
   - imuGyrNoise: Increase if orientation drifts too fast
   - imuAccBiasN: Decrease if acceleration bias changes slowly
   - imuGyrBiasN: Decrease if gyroscope bias changes slowly

3. Covariance Thresholds:
   - priorPoseNoise: Initial pose uncertainty (default: 1e-2)
   - priorVelNoise: Initial velocity uncertainty (default: 1e4)
   - correctionNoise: LiDAR-IMU correction (default: [0.05, 0.1])

4. Feature Matching:
   - edgeThreshold: Lower for more edge features
   - surfThreshold: Lower for more surface features
   - mappingCornerLeafSize: Lower for denser map but slower
""")

        print()

    def plot_covariance_curves(self):
        """绘制协方差曲线"""
        print("="*60)
        print("Generating Covariance Analysis Plots")
        print("="*60)

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)

        # 1. 位置误差演化 - 使用时间对齐后的数据
        ax1 = fig.add_subplot(gs[0, 0])
        aligned = self.analysis_results.get('aligned_trajectory')
        if aligned is not None:
            times = aligned['times']
            time = (times - times[0]) / 1e9  # Convert to seconds from start

            dx = aligned['dx']
            dy = aligned['dy']
            dz = aligned['dz']

            ax1.plot(time, dx, 'r-', alpha=0.7, label='X error')
            ax1.plot(time, dy, 'g-', alpha=0.7, label='Y error')
            ax1.plot(time, dz, 'b-', alpha=0.7, label='Z error')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position Error (m)')
            ax1.set_title('GPS-Fusion Position Error (Time-Aligned)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 添加误差统计信息
            error_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            ax1.axhline(y=np.mean(error_3d), color='k', linestyle='--', alpha=0.5,
                       label=f'Mean 3D: {np.mean(error_3d):.2f}m')
        else:
            ax1.text(0.5, 0.5, 'No aligned trajectory data',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('GPS-Fusion Position Error (Time-Aligned)')

        # 2. 速度异常分布
        ax2 = fig.add_subplot(gs[0, 1])
        if self.diagnostic_data is not None:
            vel_anomalies = self.diagnostic_data[
                self.diagnostic_data['type'] == 'VEL_ANOMALY'
            ]['value'].values

            if len(vel_anomalies) > 0:
                ax2.hist(vel_anomalies, bins=50, alpha=0.7, color='orange', edgecolor='black')
                ax2.axvline(x=np.mean(vel_anomalies), color='r', linestyle='--',
                           label=f'Mean: {np.mean(vel_anomalies):.2f}')
                ax2.set_xlabel('Velocity (m/s)')
                ax2.set_ylabel('Count')
                ax2.set_title('Velocity Anomaly Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

        # 3. IMU加速度
        ax3 = fig.add_subplot(gs[1, 0])
        if self.imu_acceleration is not None:
            acc = self.imu_acceleration
            time = acc['relative_time_s'].values

            ax3.plot(time, acc['ax_m_s2'].values, 'r-', alpha=0.5, label='X')
            ax3.plot(time, acc['ay_m_s2'].values, 'g-', alpha=0.5, label='Y')
            ax3.plot(time, acc['az_m_s2'].values, 'b-', alpha=0.5, label='Z')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Acceleration (m/s^2)')
            ax3.set_title('IMU Linear Acceleration')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. IMU角速度
        ax4 = fig.add_subplot(gs[1, 1])
        if self.imu_angular_velocity is not None:
            gyro = self.imu_angular_velocity
            time = gyro['relative_time_s'].values

            ax4.plot(time, np.rad2deg(gyro['wx_rad_s'].values), 'r-', alpha=0.5, label='X')
            ax4.plot(time, np.rad2deg(gyro['wy_rad_s'].values), 'g-', alpha=0.5, label='Y')
            ax4.plot(time, np.rad2deg(gyro['wz_rad_s'].values), 'b-', alpha=0.5, label='Z')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Angular Velocity (deg/s)')
            ax4.set_title('IMU Angular Velocity')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. IMU姿态
        ax5 = fig.add_subplot(gs[2, 0])
        if self.imu_orientation is not None:
            ori = self.imu_orientation
            time = ori['relative_time_s'].values

            ax5.plot(time, ori['roll_deg'].values, 'r-', alpha=0.7, label='Roll')
            ax5.plot(time, ori['pitch_deg'].values, 'g-', alpha=0.7, label='Pitch')
            ax5.plot(time, ori['yaw_deg'].values, 'b-', alpha=0.7, label='Yaw')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Orientation (deg)')
            ax5.set_title('IMU Orientation (RPY)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 延迟分析
        ax6 = fig.add_subplot(gs[2, 1])
        if self.gps_latency is not None and self.fusion_latency is not None:
            gps_lat = self.gps_latency
            fusion_lat = self.fusion_latency

            ax6.hist(gps_lat['latency_ms'].values, bins=30, alpha=0.6,
                    label=f'GPS (mean: {np.mean(gps_lat["latency_ms"]):.1f}ms)', color='blue')
            ax6.hist(fusion_lat['latency_ms'].values, bins=30, alpha=0.6,
                    label=f'Fusion (mean: {np.mean(fusion_lat["latency_ms"]):.1f}ms)', color='orange')
            ax6.set_xlabel('Latency (ms)')
            ax6.set_ylabel('Count')
            ax6.set_title('Latency Distribution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(self.output_dir, 'covariance_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_gps_trajectory_comparison(self):
        """绘制GPS轨迹对比图 (左/右/合并) - 使用时间对齐数据"""
        print("="*60)
        print("Generating GPS Trajectory Comparison Plots")
        print("="*60)

        fig = plt.figure(figsize=(20, 12))

        # 1. GPS轨迹 (左上)
        ax1 = fig.add_subplot(231)
        if self.gps_trajectory is not None:
            gps = self.gps_trajectory
            scatter = ax1.scatter(gps['x'].values, gps['y'].values,
                                 c=range(len(gps)), cmap='viridis', s=10, alpha=0.7)
            ax1.plot(gps['x'].values[0], gps['y'].values[0], 'go', markersize=15, label='Start')
            ax1.plot(gps['x'].values[-1], gps['y'].values[-1], 'ro', markersize=15, label='End')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('GPS Trajectory (Raw)')
            ax1.legend()
            ax1.axis('equal')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Time index')

        # 2. 融合轨迹 (中上)
        ax2 = fig.add_subplot(232)
        if self.fusion_trajectory is not None:
            fusion = self.fusion_trajectory
            scatter = ax2.scatter(fusion['x'].values, fusion['y'].values,
                                 c=range(len(fusion)), cmap='plasma', s=10, alpha=0.7)
            ax2.plot(fusion['x'].values[0], fusion['y'].values[0], 'go', markersize=15, label='Start')
            ax2.plot(fusion['x'].values[-1], fusion['y'].values[-1], 'ro', markersize=15, label='End')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('LIO-SAM Fusion Trajectory (Raw)')
            ax2.legend()
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Time index')

        # 3. 原始对比 (右上) - 未对齐
        ax3 = fig.add_subplot(233)
        if self.gps_trajectory is not None and self.fusion_trajectory is not None:
            gps = self.gps_trajectory
            fusion = self.fusion_trajectory

            ax3.plot(gps['x'].values, gps['y'].values, 'b-', alpha=0.7, linewidth=1.5, label='GPS')
            ax3.plot(fusion['x'].values, fusion['y'].values, 'r-', alpha=0.7, linewidth=1.5, label='Fusion')

            ax3.plot(gps['x'].values[0], gps['y'].values[0], 'bs', markersize=10)
            ax3.plot(fusion['x'].values[0], fusion['y'].values[0], 'rs', markersize=10)
            ax3.plot(gps['x'].values[-1], gps['y'].values[-1], 'b^', markersize=10)
            ax3.plot(fusion['x'].values[-1], fusion['y'].values[-1], 'r^', markersize=10)

            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_title('Raw Comparison (Not Aligned)')
            ax3.legend()
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)

        # 4. 时间对齐后的GPS插值轨迹 (左下)
        ax4 = fig.add_subplot(234)
        aligned = self.analysis_results.get('aligned_trajectory')
        if aligned is not None:
            gps_x = aligned['gps_x']
            gps_y = aligned['gps_y']
            scatter = ax4.scatter(gps_x, gps_y, c=range(len(gps_x)), cmap='viridis', s=10, alpha=0.7)
            ax4.plot(gps_x[0], gps_y[0], 'go', markersize=15, label='Start')
            ax4.plot(gps_x[-1], gps_y[-1], 'ro', markersize=15, label='End')
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            ax4.set_title('GPS Trajectory (Time-Aligned)')
            ax4.legend()
            ax4.axis('equal')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Time index')

        # 5. 时间对齐后的Fusion轨迹 (中下)
        ax5 = fig.add_subplot(235)
        if aligned is not None:
            fusion_x = aligned['fusion_x']
            fusion_y = aligned['fusion_y']
            scatter = ax5.scatter(fusion_x, fusion_y, c=range(len(fusion_x)), cmap='plasma', s=10, alpha=0.7)
            ax5.plot(fusion_x[0], fusion_y[0], 'go', markersize=15, label='Start')
            ax5.plot(fusion_x[-1], fusion_y[-1], 'ro', markersize=15, label='End')
            ax5.set_xlabel('X (m)')
            ax5.set_ylabel('Y (m)')
            ax5.set_title('Fusion Trajectory (Time-Aligned)')
            ax5.legend()
            ax5.axis('equal')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='Time index')

        # 6. 时间对齐后的对比 (右下) - 关键图
        ax6 = fig.add_subplot(236)
        if aligned is not None:
            gps_x = aligned['gps_x']
            gps_y = aligned['gps_y']
            fusion_x = aligned['fusion_x']
            fusion_y = aligned['fusion_y']

            ax6.plot(gps_x, gps_y, 'b-', alpha=0.8, linewidth=2, label='GPS (interpolated)')
            ax6.plot(fusion_x, fusion_y, 'r-', alpha=0.8, linewidth=2, label='Fusion')

            # 绘制误差线（每隔N个点）
            skip = max(1, len(gps_x) // 30)
            for i in range(0, len(gps_x), skip):
                ax6.plot([gps_x[i], fusion_x[i]], [gps_y[i], fusion_y[i]],
                        'g-', alpha=0.3, linewidth=0.5)

            # 标记起止点
            ax6.plot(gps_x[0], gps_y[0], 'bs', markersize=12, label='GPS Start')
            ax6.plot(fusion_x[0], fusion_y[0], 'rs', markersize=12, label='Fusion Start')
            ax6.plot(gps_x[-1], gps_y[-1], 'b^', markersize=12, label='GPS End')
            ax6.plot(fusion_x[-1], fusion_y[-1], 'r^', markersize=12, label='Fusion End')

            # 计算并显示统计信息
            error_2d = np.sqrt(aligned['dx']**2 + aligned['dy']**2)
            end_err = np.sqrt((fusion_x[-1]-gps_x[-1])**2 + (fusion_y[-1]-gps_y[-1])**2)

            ax6.set_xlabel('X (m)')
            ax6.set_ylabel('Y (m)')
            ax6.set_title(f'Time-Aligned Comparison\nMean 2D Error: {np.mean(error_2d):.2f}m, End Error: {end_err:.2f}m')
            ax6.legend(loc='upper left', fontsize=8)
            ax6.axis('equal')
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(self.output_dir, 'gps_trajectory_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def plot_diagnostic_timeline(self):
        """绘制诊断时间线"""
        if self.diagnostic_data is None or len(self.diagnostic_data) == 0:
            print("No diagnostic data for timeline plot")
            return

        print("="*60)
        print("Generating Diagnostic Timeline Plot")
        print("="*60)

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        df = self.diagnostic_data

        # 归一化时间戳
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            t0 = timestamps.min()
            relative_time = timestamps - t0
        else:
            relative_time = np.arange(len(df))

        # 1. 位置跳变事件
        ax1 = axes[0]
        pos_jumps = df[df['type'] == 'POS_JUMP']
        if len(pos_jumps) > 0:
            pos_times = pos_jumps['timestamp'].values - t0
            pos_values = pos_jumps['value'].values
            ax1.scatter(pos_times, pos_values, c='red', s=30, alpha=0.7)
            ax1.set_ylabel('Implied Velocity (m/s)')
            ax1.set_title('Position Jump Events')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)

        # 2. 速度异常事件
        ax2 = axes[1]
        vel_anomalies = df[df['type'] == 'VEL_ANOMALY']
        if len(vel_anomalies) > 0:
            vel_times = vel_anomalies['timestamp'].values - t0
            vel_values = vel_anomalies['value'].values
            ax2.plot(vel_times, vel_values, 'b-', alpha=0.5, linewidth=0.5)
            ax2.scatter(vel_times, vel_values, c='blue', s=10, alpha=0.5)
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('Velocity Anomaly Events')
            ax2.grid(True, alpha=0.3)

        # 3. 按来源分类的事件数量
        ax3 = axes[2]
        sources = df['source'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(sources)))

        for source, color in zip(sources, colors):
            source_data = df[df['source'] == source]
            if len(source_data) > 0:
                times = source_data['timestamp'].values - t0
                ax3.scatter(times, [source]*len(times), c=[color], s=20, alpha=0.6, label=source)

        ax3.set_xlabel('Relative Time (s)')
        ax3.set_ylabel('Source')
        ax3.set_title('Events by Source')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        output_path = os.path.join(self.output_dir, 'diagnostic_timeline.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    def generate_report(self):
        """生成综合分析报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.output_dir, f'diagnostic_report_{timestamp}.md')

        with open(report_path, 'w') as f:
            f.write("# LIO-SAM Diagnostic Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 健康度评分
            health_score = self.analysis_results.get('health_score', 'N/A')
            f.write("## System Health Score\n\n")
            f.write(f"**Score: {health_score}/100**\n\n")

            issues = self.analysis_results.get('health_issues', [])
            if issues:
                f.write("### Issues Detected:\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
                f.write("\n")

            # 异常统计
            f.write("## Anomaly Statistics\n\n")
            anomaly_stats = self.analysis_results.get('anomaly_stats', {})
            if anomaly_stats:
                f.write("| Type | Count | Mean | Std | Min | Max |\n")
                f.write("|------|-------|------|-----|-----|-----|\n")
                for event_type, stats in anomaly_stats.items():
                    f.write(f"| {event_type} | {stats['count']} | {stats['mean']:.2f} | "
                           f"{stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n")
                f.write("\n")

            # GPS-Fusion误差
            gps_error = self.analysis_results.get('gps_fusion_error', {})
            if gps_error:
                f.write("## GPS-Fusion Error (Time-Aligned)\n\n")
                f.write("### 3D Error\n")
                f.write(f"- Mean: {gps_error.get('mean', 'N/A'):.3f} m\n")
                f.write(f"- Std: {gps_error.get('std', 'N/A'):.3f} m\n")
                f.write(f"- Max: {gps_error.get('max', 'N/A'):.3f} m\n\n")
                f.write("### 2D Error (XY plane)\n")
                f.write(f"- Mean: {gps_error.get('mean_2d', 'N/A'):.3f} m\n")
                f.write(f"- Std: {gps_error.get('std_2d', 'N/A'):.3f} m\n")
                f.write(f"- Max: {gps_error.get('max_2d', 'N/A'):.3f} m\n\n")
                f.write("### End Point Error\n")
                f.write(f"- 3D: {gps_error.get('end_error', 'N/A'):.3f} m\n")
                f.write(f"- 2D: {gps_error.get('end_error_2d', 'N/A'):.3f} m\n\n")

            # 优化建议
            f.write("## Optimization Suggestions\n\n")
            if self.optimization_suggestions:
                for i, suggestion in enumerate(self.optimization_suggestions, 1):
                    f.write(f"{i}. {suggestion}\n\n")
            else:
                f.write("No critical issues detected.\n\n")

            # 参数调优指南
            f.write("## Parameter Tuning Guide\n\n")
            f.write("""
### GPS Weight Control (params.yaml)
- `gpsCovThreshold`: Increase to accept more GPS data (current: 5.0)
- `poseCovThreshold`: Increase to add GPS factors more frequently (current: 25.0)
- `gpsNoiseMin`: Decrease for higher GPS weight (current: 0.5)
- `gpsNoiseScale`: Decrease for higher GPS weight (current: 0.5)
- `gpsAddInterval`: Decrease for more frequent GPS factors (current: 3.0)

### IMU Parameters (params.yaml)
- `imuAccNoise`: Accelerometer white noise
- `imuGyrNoise`: Gyroscope white noise
- `imuAccBiasN`: Accelerometer bias random walk
- `imuGyrBiasN`: Gyroscope bias random walk
- `imuGravity`: Local gravity magnitude

### Key Covariance Thresholds
- GPS Factor is added when: `poseCovariance(3,3) >= poseCovThreshold` OR `poseCovariance(4,4) >= poseCovThreshold`
- GPS data is rejected when: `noise_x > gpsCovThreshold` OR `noise_y > gpsCovThreshold`
""")

        print(f"\nReport saved: {report_path}")
        return report_path

    def run_full_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*60)
        print("LIO-SAM Comprehensive Diagnostic Analysis")
        print("="*60 + "\n")

        # 加载数据
        self.load_all_data()

        # 分析
        self.analyze_diagnostic_events()
        self.analyze_covariance_evolution()
        self.analyze_gps_status()
        self.analyze_imu_preintegration()
        self.analyze_factor_graph_health()
        self.generate_optimization_suggestions()

        # 生成可视化
        self.plot_covariance_curves()
        self.plot_gps_trajectory_comparison()
        self.plot_diagnostic_timeline()

        # 生成报告
        report_path = self.generate_report()

        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        print(f"\nOutput files saved to: {self.output_dir}")
        print("  - covariance_analysis.png")
        print("  - gps_trajectory_comparison.png")
        print("  - diagnostic_timeline.png")
        print(f"  - {os.path.basename(report_path)}")


def main():
    parser = argparse.ArgumentParser(description='LIO-SAM Comprehensive Diagnostic Analyzer')
    parser.add_argument('--diagnostic-csv', '-d', type=str,
                        help='Path to diagnostic CSV file')
    parser.add_argument('--output-dir', '-o', type=str,
                        default='/root/autodl-tmp/catkin_ws/src/LIO-SAM/output',
                        help='Output directory for analysis results')
    parser.add_argument('--latest', '-l', action='store_true',
                        help='Use the latest diagnostic CSV file from /tmp')

    args = parser.parse_args()

    diagnostic_csv = args.diagnostic_csv

    # 如果指定使用最新的诊断文件
    if args.latest:
        import glob
        csv_files = glob.glob('/tmp/lio_sam_diagnostic_*.csv')
        if csv_files:
            diagnostic_csv = max(csv_files, key=os.path.getmtime)
            print(f"Using latest diagnostic file: {diagnostic_csv}")
        else:
            print("No diagnostic files found in /tmp")

    analyzer = LIOSAMDiagnosticAnalyzer(
        diagnostic_csv=diagnostic_csv,
        output_dir=args.output_dir
    )

    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
