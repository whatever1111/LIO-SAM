#!/usr/bin/env python3
"""
LIO-SAM Diagnostic Monitor
监控 IMU、LiDAR 和 LIO-SAM 输出，诊断异常高速度问题
"""

import rospy
import numpy as np
import sys
import signal
import threading
import time
from collections import deque
from datetime import datetime
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans

class DiagnosticMonitor:
    def __init__(self):
        rospy.init_node('diagnostic_monitor', anonymous=True)

        # 配置参数
        self.window_size = 100  # 滑动窗口大小
        self.velocity_threshold = 10.0  # m/s，正常速度阈值
        self.acc_threshold = 50.0  # m/s^2，正常加速度阈值
        self.angular_rate_threshold = 10.0  # rad/s，正常角速度阈值

        # 数据缓冲区
        self.imu_buffer = deque(maxlen=self.window_size)
        self.lidar_buffer = deque(maxlen=self.window_size)
        self.odom_buffer = deque(maxlen=self.window_size)
        self.preint_buffer = deque(maxlen=self.window_size)

        # 统计信息
        self.imu_stats = {'count': 0, 'anomalies': 0, 'last_time': None}
        self.lidar_stats = {'count': 0, 'anomalies': 0, 'last_time': None}
        self.odom_stats = {'count': 0, 'anomalies': 0, 'last_time': None}
        self.preint_stats = {'count': 0, 'anomalies': 0, 'last_time': None}

        # 异常事件记录
        self.anomaly_events = []

        # 输出文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'/tmp/lio_sam_diagnostic_{timestamp}.txt'
        self.csv_file = f'/tmp/lio_sam_diagnostic_{timestamp}.csv'

        # 初始化CSV文件
        with open(self.csv_file, 'w') as f:
            f.write("timestamp,source,type,value,description\n")

        # 订阅器
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=100)
        self.lidar_sub = rospy.Subscriber('/lidar_points', PointCloud2, self.lidar_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber('lio_sam/mapping/odometry', Odometry, self.odom_callback, queue_size=100)
        self.preint_sub = rospy.Subscriber('odometry/imu', Odometry, self.preint_callback, queue_size=100)

        # 状态监控线程
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        self.log_message("=== LIO-SAM Diagnostic Monitor Started ===")
        self.log_message(f"Log file: {self.log_file}")
        self.log_message(f"CSV file: {self.csv_file}")

    def _extract_xyz_fast(self, msg):
        """
        Fast PointCloud2 XYZ extraction using numpy.frombuffer.
        Falls back to the python generator if the layout is unexpected.
        Returns an (N,3) float array (may be empty).
        """
        try:
            num_points = int(msg.width) * int(msg.height)
            if num_points <= 0 or msg.point_step <= 0:
                return np.empty((0, 3), dtype=np.float32)

            field_map = {f.name: f for f in msg.fields}
            for name in ("x", "y", "z"):
                if name not in field_map:
                    raise ValueError(f"missing field {name}")
                if int(field_map[name].datatype) != 7:  # FLOAT32
                    raise ValueError(f"field {name} not FLOAT32")

            endian = ">" if msg.is_bigendian else "<"
            dtype = np.dtype(
                {
                    "names": ["x", "y", "z"],
                    "formats": [endian + "f4", endian + "f4", endian + "f4"],
                    "offsets": [field_map["x"].offset, field_map["y"].offset, field_map["z"].offset],
                    "itemsize": msg.point_step,
                }
            )
            arr = np.frombuffer(msg.data, dtype=dtype, count=num_points)
            xyz = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)
            mask = np.isfinite(xyz).all(axis=1)
            if not mask.any():
                return np.empty((0, 3), dtype=np.float32)
            return xyz[mask]
        except Exception:
            # Fallback: slower but more compatible
            pts = []
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append(p)
            if not pts:
                return np.empty((0, 3), dtype=np.float32)
            return np.array(pts, dtype=np.float32)

    def log_message(self, msg):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def log_csv(self, source, event_type, value, description):
        """记录CSV数据"""
        timestamp = rospy.Time.now().to_sec()
        with open(self.csv_file, 'a') as f:
            f.write(f"{timestamp},{source},{event_type},{value},\"{description}\"\n")

    def imu_callback(self, msg):
        """IMU数据回调"""
        self.imu_stats['count'] += 1

        # 提取数据
        acc = np.array([msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z])

        acc_norm = np.linalg.norm(acc)
        gyro_norm = np.linalg.norm(gyro)

        # 检查异常
        anomaly_detected = False

        # 检查加速度异常
        if acc_norm > self.acc_threshold:
            anomaly_detected = True
            self.imu_stats['anomalies'] += 1
            msg_text = f"IMU Acceleration Anomaly: norm={acc_norm:.2f} m/s^2, x={acc[0]:.2f}, y={acc[1]:.2f}, z={acc[2]:.2f}"
            self.log_message(f"[WARNING] {msg_text}")
            self.log_csv("IMU", "ACC_ANOMALY", acc_norm, msg_text)

        # 检查角速度异常
        if gyro_norm > self.angular_rate_threshold:
            anomaly_detected = True
            self.imu_stats['anomalies'] += 1
            msg_text = f"IMU Angular Rate Anomaly: norm={gyro_norm:.2f} rad/s, x={gyro[0]:.2f}, y={gyro[1]:.2f}, z={gyro[2]:.2f}"
            self.log_message(f"[WARNING] {msg_text}")
            self.log_csv("IMU", "GYRO_ANOMALY", gyro_norm, msg_text)

        # 检查数据有效性
        if np.any(np.isnan(acc)) or np.any(np.isnan(gyro)):
            anomaly_detected = True
            msg_text = "IMU data contains NaN values"
            self.log_message(f"[ERROR] {msg_text}")
            self.log_csv("IMU", "NAN_VALUE", 0, msg_text)

        # 记录到缓冲区
        self.imu_buffer.append({
            'time': msg.header.stamp.to_sec(),
            'acc': acc,
            'gyro': gyro,
            'anomaly': anomaly_detected
        })

        self.imu_stats['last_time'] = msg.header.stamp.to_sec()

    def lidar_callback(self, msg):
        """LiDAR点云数据回调"""
        self.lidar_stats['count'] += 1

        # 提取点云信息（使用 numpy 快速路径，避免 Python 逐点遍历造成大量 CPU 开销）
        xyz = self._extract_xyz_fast(msg)
        num_points = int(xyz.shape[0])

        if num_points > 0:
            ranges = np.linalg.norm(xyz, axis=1)
            min_range = float(np.min(ranges))
            max_range = float(np.max(ranges))
            mean_range = float(np.mean(ranges))

            # 检查异常
            anomaly_detected = False

            # 检查点云数量异常
            if num_points < 100:  # 点云太少
                anomaly_detected = True
                self.lidar_stats['anomalies'] += 1
                msg_text = f"LiDAR point cloud too sparse: {num_points} points"
                self.log_message(f"[WARNING] {msg_text}")
                self.log_csv("LIDAR", "SPARSE_CLOUD", num_points, msg_text)

            # 检查距离异常
            if max_range > 200:  # 超过200米的点
                anomaly_detected = True
                self.lidar_stats['anomalies'] += 1
                msg_text = f"LiDAR max range anomaly: {max_range:.2f} m"
                self.log_message(f"[WARNING] {msg_text}")
                self.log_csv("LIDAR", "RANGE_ANOMALY", max_range, msg_text)

            # 记录到缓冲区
            self.lidar_buffer.append({
                'time': msg.header.stamp.to_sec(),
                'num_points': num_points,
                'min_range': min_range,
                'max_range': max_range,
                'mean_range': mean_range,
                'anomaly': anomaly_detected
            })

        else:
            msg_text = "LiDAR received empty point cloud"
            self.log_message(f"[ERROR] {msg_text}")
            self.log_csv("LIDAR", "EMPTY_CLOUD", 0, msg_text)

        self.lidar_stats['last_time'] = msg.header.stamp.to_sec()

    def odom_callback(self, msg):
        """LIO-SAM里程计输出回调"""
        self.odom_stats['count'] += 1

        # 提取速度
        vel = np.array([msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y,
                       msg.twist.twist.linear.z])
        vel_norm = np.linalg.norm(vel)

        # 提取位置
        pos = np.array([msg.pose.pose.position.x,
                       msg.pose.pose.position.y,
                       msg.pose.pose.position.z])

        # 检查异常
        anomaly_detected = False

        # 检查速度异常
        if vel_norm > self.velocity_threshold:
            anomaly_detected = True
            self.odom_stats['anomalies'] += 1
            msg_text = f"LIO-SAM Velocity Anomaly: norm={vel_norm:.2f} m/s, vx={vel[0]:.2f}, vy={vel[1]:.2f}, vz={vel[2]:.2f}"
            self.log_message(f"[ERROR] {msg_text}")
            self.log_csv("LIO_ODOM", "VEL_ANOMALY", vel_norm, msg_text)

            # 记录异常事件详情
            self.anomaly_events.append({
                'time': msg.header.stamp.to_sec(),
                'type': 'LIO_VEL_ANOMALY',
                'velocity': vel,
                'position': pos
            })

        # 检查位置跳变
        if len(self.odom_buffer) > 0:
            last_pos = self.odom_buffer[-1]['pos']
            dt = msg.header.stamp.to_sec() - self.odom_buffer[-1]['time']
            if dt > 0:
                pos_change = np.linalg.norm(pos - last_pos)
                implied_vel = pos_change / dt
                if implied_vel > self.velocity_threshold * 2:
                    msg_text = f"LIO-SAM Position Jump: {pos_change:.2f}m in {dt:.3f}s (implied vel: {implied_vel:.2f} m/s)"
                    self.log_message(f"[ERROR] {msg_text}")
                    self.log_csv("LIO_ODOM", "POS_JUMP", implied_vel, msg_text)

        # 记录到缓冲区
        self.odom_buffer.append({
            'time': msg.header.stamp.to_sec(),
            'pos': pos,
            'vel': vel,
            'anomaly': anomaly_detected
        })

        self.odom_stats['last_time'] = msg.header.stamp.to_sec()

    def preint_callback(self, msg):
        """IMU预积分输出回调"""
        self.preint_stats['count'] += 1

        # 提取速度
        vel = np.array([msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y,
                       msg.twist.twist.linear.z])
        vel_norm = np.linalg.norm(vel)

        # 检查异常
        anomaly_detected = False

        # 检查预积分速度异常（这是最直接相关的）
        if vel_norm > self.velocity_threshold:
            anomaly_detected = True
            self.preint_stats['anomalies'] += 1
            msg_text = f"IMU Preintegration Velocity Anomaly: norm={vel_norm:.2f} m/s, vx={vel[0]:.2f}, vy={vel[1]:.2f}, vz={vel[2]:.2f}"
            self.log_message(f"[CRITICAL] {msg_text}")
            self.log_csv("IMU_PREINT", "VEL_ANOMALY", vel_norm, msg_text)

            # 分析可能原因
            self.analyze_anomaly_cause(msg.header.stamp.to_sec())

        # 记录到缓冲区
        self.preint_buffer.append({
            'time': msg.header.stamp.to_sec(),
            'vel': vel,
            'anomaly': anomaly_detected
        })

        self.preint_stats['last_time'] = msg.header.stamp.to_sec()

    def analyze_anomaly_cause(self, anomaly_time):
        """分析异常原因"""
        self.log_message("=== Analyzing Anomaly Cause ===")

        # 检查IMU数据
        recent_imu = [d for d in self.imu_buffer if abs(d['time'] - anomaly_time) < 1.0]
        if recent_imu:
            imu_anomalies = sum(1 for d in recent_imu if d['anomaly'])
            self.log_message(f"  IMU: {len(recent_imu)} samples, {imu_anomalies} anomalies")
            if imu_anomalies > 0:
                self.log_message("  -> IMU data shows anomalies, likely IMU sensor issue")

        # 检查LiDAR数据
        recent_lidar = [d for d in self.lidar_buffer if abs(d['time'] - anomaly_time) < 1.0]
        if recent_lidar:
            lidar_anomalies = sum(1 for d in recent_lidar if d['anomaly'])
            self.log_message(f"  LiDAR: {len(recent_lidar)} samples, {lidar_anomalies} anomalies")
            if lidar_anomalies > 0:
                self.log_message("  -> LiDAR data shows anomalies, possible sensor or environment issue")

        # 检查时间同步
        if recent_imu and recent_lidar:
            imu_times = [d['time'] for d in recent_imu]
            lidar_times = [d['time'] for d in recent_lidar]
            time_diff = np.mean(lidar_times) - np.mean(imu_times)
            self.log_message(f"  Time sync: IMU-LiDAR diff = {time_diff:.3f}s")
            if abs(time_diff) > 0.1:
                self.log_message("  -> Large time difference detected, possible synchronization issue")

        self.log_message("================================")

    def monitor_loop(self):
        """监控循环，定期输出统计信息"""
        rate = rospy.Rate(1)  # 1 Hz

        while not rospy.is_shutdown():
            # 每秒输出统计
            self.print_statistics()

            # 检查数据流健康状况
            self.check_data_health()

            rate.sleep()

    def print_statistics(self):
        """打印统计信息"""
        current_time = rospy.Time.now().to_sec()

        # 清屏效果
        print("\n" + "="*60)
        print(f"LIO-SAM Diagnostic Monitor - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)

        # IMU统计
        imu_rate = self.calculate_rate(self.imu_stats)
        print(f"IMU:        {self.imu_stats['count']:6d} msgs | "
              f"{imu_rate:5.1f} Hz | "
              f"{self.imu_stats['anomalies']:4d} anomalies")

        # LiDAR统计
        lidar_rate = self.calculate_rate(self.lidar_stats)
        print(f"LiDAR:      {self.lidar_stats['count']:6d} msgs | "
              f"{lidar_rate:5.1f} Hz | "
              f"{self.lidar_stats['anomalies']:4d} anomalies")

        # 里程计统计
        odom_rate = self.calculate_rate(self.odom_stats)
        print(f"LIO Odom:   {self.odom_stats['count']:6d} msgs | "
              f"{odom_rate:5.1f} Hz | "
              f"{self.odom_stats['anomalies']:4d} anomalies")

        # 预积分统计
        preint_rate = self.calculate_rate(self.preint_stats)
        print(f"IMU Preint: {self.preint_stats['count']:6d} msgs | "
              f"{preint_rate:5.1f} Hz | "
              f"{self.preint_stats['anomalies']:4d} anomalies")

        # 显示最近的异常
        if self.anomaly_events:
            print("\nRecent Anomalies:")
            for event in self.anomaly_events[-3:]:  # 显示最近3个
                print(f"  [{event['type']}] vel={np.linalg.norm(event['velocity']):.2f} m/s")

    def calculate_rate(self, stats):
        """计算数据频率"""
        if stats['last_time'] is None or stats['count'] < 2:
            return 0.0

        # 使用最近的时间窗口计算频率
        window = min(stats['count'], 100)
        if hasattr(self, 'start_time'):
            elapsed = stats['last_time'] - self.start_time
            if elapsed > 0:
                return stats['count'] / elapsed
        return 0.0

    def check_data_health(self):
        """检查数据流健康状况"""
        current_time = rospy.Time.now().to_sec()

        # 检查IMU超时
        if self.imu_stats['last_time'] and current_time - self.imu_stats['last_time'] > 1.0:
            self.log_message("[WARNING] IMU data timeout (> 1s)")

        # 检查LiDAR超时
        if self.lidar_stats['last_time'] and current_time - self.lidar_stats['last_time'] > 1.0:
            self.log_message("[WARNING] LiDAR data timeout (> 1s)")

    def shutdown(self):
        """关闭时的清理工作"""
        self.log_message("\n=== Diagnostic Monitor Shutdown ===")
        self.log_message(f"Total IMU anomalies: {self.imu_stats['anomalies']}")
        self.log_message(f"Total LiDAR anomalies: {self.lidar_stats['anomalies']}")
        self.log_message(f"Total LIO Odom anomalies: {self.odom_stats['anomalies']}")
        self.log_message(f"Total IMU Preint anomalies: {self.preint_stats['anomalies']}")
        self.log_message(f"Log saved to: {self.log_file}")
        self.log_message(f"CSV saved to: {self.csv_file}")

def signal_handler(sig, frame):
    print("\n\nReceived interrupt, shutting down...")
    monitor.shutdown()
    rospy.signal_shutdown("User interrupt")
    sys.exit(0)

if __name__ == '__main__':
    # 记录启动时间
    monitor = DiagnosticMonitor()
    monitor.start_time = rospy.Time.now().to_sec()

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("\nMonitoring started. Press Ctrl+C to stop.\n")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        monitor.shutdown()
