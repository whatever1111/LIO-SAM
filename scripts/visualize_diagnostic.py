#!/usr/bin/env python3
"""
实时可视化诊断数据
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import sys
import os
from collections import deque
from datetime import datetime

class DiagnosticVisualizer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 10))
        self.fig.suptitle('LIO-SAM Real-time Diagnostic Visualization', fontsize=16)

        # 数据缓冲区
        self.time_window = 30  # 显示最近30秒的数据
        self.data_buffer = {
            'time': deque(maxlen=1000),
            'imu_acc': deque(maxlen=1000),
            'imu_gyro': deque(maxlen=1000),
            'lidar_points': deque(maxlen=1000),
            'velocity': deque(maxlen=1000),
            'anomaly_times': deque(maxlen=100),
            'anomaly_types': deque(maxlen=100)
        }

        # 初始化图表
        self.setup_plots()

        # 上次读取的行数
        self.last_row = 0

    def setup_plots(self):
        """初始化各个子图"""
        # 1. IMU加速度图
        self.ax_imu_acc = self.axes[0, 0]
        self.ax_imu_acc.set_title('IMU Acceleration')
        self.ax_imu_acc.set_xlabel('Time (s)')
        self.ax_imu_acc.set_ylabel('Acceleration (m/s²)')
        self.ax_imu_acc.grid(True, alpha=0.3)
        self.line_imu_acc, = self.ax_imu_acc.plot([], [], 'b-', label='Acc Norm')
        self.ax_imu_acc.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Threshold')
        self.ax_imu_acc.legend(loc='upper right')

        # 2. IMU角速度图
        self.ax_imu_gyro = self.axes[0, 1]
        self.ax_imu_gyro.set_title('IMU Gyroscope')
        self.ax_imu_gyro.set_xlabel('Time (s)')
        self.ax_imu_gyro.set_ylabel('Angular Rate (rad/s)')
        self.ax_imu_gyro.grid(True, alpha=0.3)
        self.line_imu_gyro, = self.ax_imu_gyro.plot([], [], 'g-', label='Gyro Norm')
        self.ax_imu_gyro.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Threshold')
        self.ax_imu_gyro.legend(loc='upper right')

        # 3. LiDAR点云数量
        self.ax_lidar = self.axes[1, 0]
        self.ax_lidar.set_title('LiDAR Point Cloud')
        self.ax_lidar.set_xlabel('Time (s)')
        self.ax_lidar.set_ylabel('Number of Points')
        self.ax_lidar.grid(True, alpha=0.3)
        self.line_lidar, = self.ax_lidar.plot([], [], 'c-', label='Points')
        self.ax_lidar.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Min Threshold')
        self.ax_lidar.legend(loc='upper right')

        # 4. 速度图（最关键的指标）
        self.ax_velocity = self.axes[1, 1]
        self.ax_velocity.set_title('Estimated Velocity (Key Metric)')
        self.ax_velocity.set_xlabel('Time (s)')
        self.ax_velocity.set_ylabel('Velocity (m/s)')
        self.ax_velocity.grid(True, alpha=0.3)
        self.line_velocity, = self.ax_velocity.plot([], [], 'r-', linewidth=2, label='Velocity')
        self.ax_velocity.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Warning')
        self.ax_velocity.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Critical')
        self.ax_velocity.legend(loc='upper right')

        # 5. 异常事件时间线
        self.ax_timeline = self.axes[2, 0]
        self.ax_timeline.set_title('Anomaly Timeline')
        self.ax_timeline.set_xlabel('Time (s)')
        self.ax_timeline.set_ylabel('Anomaly Type')
        self.ax_timeline.grid(True, alpha=0.3)
        self.scatter_anomalies = self.ax_timeline.scatter([], [], c=[], s=50, cmap='hot')

        # 6. 统计信息
        self.ax_stats = self.axes[2, 1]
        self.ax_stats.set_title('Statistics')
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', transform=self.ax_stats.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            fontfamily='monospace')

        plt.tight_layout()

    def read_new_data(self):
        """读取CSV文件中的新数据"""
        try:
            # 读取CSV文件
            df = pd.read_csv(self.csv_file)

            if len(df) > self.last_row:
                # 获取新数据
                new_data = df.iloc[self.last_row:]
                self.last_row = len(df)

                # 处理新数据
                for _, row in new_data.iterrows():
                    timestamp = row['timestamp']
                    source = row['source']
                    event_type = row['type']
                    value = row['value']

                    # 更新时间
                    self.data_buffer['time'].append(timestamp)

                    # 根据事件类型更新相应的数据
                    if source == 'IMU' and event_type == 'ACC_ANOMALY':
                        self.data_buffer['imu_acc'].append((timestamp, value))
                    elif source == 'IMU' and event_type == 'GYRO_ANOMALY':
                        self.data_buffer['imu_gyro'].append((timestamp, value))
                    elif source == 'LIDAR' and event_type == 'SPARSE_CLOUD':
                        self.data_buffer['lidar_points'].append((timestamp, value))
                    elif 'VEL_ANOMALY' in event_type:
                        self.data_buffer['velocity'].append((timestamp, value))
                        self.data_buffer['anomaly_times'].append(timestamp)
                        self.data_buffer['anomaly_types'].append(event_type)

                return True
        except Exception as e:
            print(f"Error reading data: {e}")
            return False

        return False

    def update_plots(self, frame):
        """更新图表"""
        # 读取新数据
        self.read_new_data()

        # 获取当前时间范围
        if self.data_buffer['time']:
            current_time = max(self.data_buffer['time'])
            min_time = current_time - self.time_window
        else:
            return self.line_imu_acc,

        # 更新IMU加速度图
        if self.data_buffer['imu_acc']:
            acc_data = [(t, v) for t, v in self.data_buffer['imu_acc'] if t >= min_time]
            if acc_data:
                times, values = zip(*acc_data)
                self.line_imu_acc.set_data(times, values)
                self.ax_imu_acc.set_xlim(min_time, current_time)
                self.ax_imu_acc.set_ylim(0, max(100, max(values) * 1.2))

        # 更新IMU角速度图
        if self.data_buffer['imu_gyro']:
            gyro_data = [(t, v) for t, v in self.data_buffer['imu_gyro'] if t >= min_time]
            if gyro_data:
                times, values = zip(*gyro_data)
                self.line_imu_gyro.set_data(times, values)
                self.ax_imu_gyro.set_xlim(min_time, current_time)
                self.ax_imu_gyro.set_ylim(0, max(20, max(values) * 1.2))

        # 更新LiDAR图
        if self.data_buffer['lidar_points']:
            lidar_data = [(t, v) for t, v in self.data_buffer['lidar_points'] if t >= min_time]
            if lidar_data:
                times, values = zip(*lidar_data)
                self.line_lidar.set_data(times, values)
                self.ax_lidar.set_xlim(min_time, current_time)
                self.ax_lidar.set_ylim(0, max(1000, max(values) * 1.2))

        # 更新速度图（最重要的）
        if self.data_buffer['velocity']:
            vel_data = [(t, v) for t, v in self.data_buffer['velocity'] if t >= min_time]
            if vel_data:
                times, values = zip(*vel_data)
                self.line_velocity.set_data(times, values)
                self.ax_velocity.set_xlim(min_time, current_time)

                # 动态调整y轴范围
                max_vel = max(values)
                if max_vel > 1000:
                    self.ax_velocity.set_ylim(0, max_vel * 1.2)
                    self.ax_velocity.set_yscale('log')
                else:
                    self.ax_velocity.set_ylim(0, max(100, max_vel * 1.2))
                    self.ax_velocity.set_yscale('linear')

        # 更新异常时间线
        if self.data_buffer['anomaly_times']:
            recent_anomalies = [(t, i) for i, t in enumerate(self.data_buffer['anomaly_times']) if t >= min_time]
            if recent_anomalies:
                times, indices = zip(*recent_anomalies)
                # 给不同类型的异常分配不同的y值
                type_map = {'VEL_ANOMALY': 1, 'ACC_ANOMALY': 2, 'GYRO_ANOMALY': 3, 'RANGE_ANOMALY': 4}
                y_values = [type_map.get(self.data_buffer['anomaly_types'][i].split('_')[0] + '_ANOMALY', 0)
                          for i in indices]
                colors = [i for i in range(len(times))]

                self.ax_timeline.clear()
                self.ax_timeline.scatter(times, y_values, c=colors, s=100, cmap='hot', alpha=0.6)
                self.ax_timeline.set_xlim(min_time, current_time)
                self.ax_timeline.set_ylim(0, 5)
                self.ax_timeline.set_yticks([1, 2, 3, 4])
                self.ax_timeline.set_yticklabels(['Velocity', 'Acceleration', 'Gyro', 'Range'])
                self.ax_timeline.set_xlabel('Time (s)')
                self.ax_timeline.set_title('Anomaly Timeline')
                self.ax_timeline.grid(True, alpha=0.3)

        # 更新统计信息
        stats_text = f"""Current Statistics:

Total Anomalies: {len(self.data_buffer['anomaly_times'])}
Recent Velocity Anomalies: {len([v for t, v in self.data_buffer['velocity'] if t >= min_time])}

Latest Values:
"""
        if self.data_buffer['velocity']:
            latest_vel = self.data_buffer['velocity'][-1][1]
            stats_text += f"  Velocity: {latest_vel:.2f} m/s\n"
            if latest_vel > 100:
                stats_text += f"  STATUS: CRITICAL! ({latest_vel/1000:.1f} km/s)\n"
            elif latest_vel > 10:
                stats_text += f"  STATUS: WARNING\n"
            else:
                stats_text += f"  STATUS: Normal\n"

        if self.data_buffer['imu_acc']:
            stats_text += f"  IMU Acc: {self.data_buffer['imu_acc'][-1][1]:.2f} m/s²\n"

        if self.data_buffer['imu_gyro']:
            stats_text += f"  IMU Gyro: {self.data_buffer['imu_gyro'][-1][1]:.2f} rad/s\n"

        self.stats_text.set_text(stats_text)

        return self.line_imu_acc, self.line_imu_gyro, self.line_lidar, self.line_velocity

    def animate(self):
        """启动动画"""
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=500, blit=False)
        plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 visualize_diagnostic.py <csv_file>")
        print("Example: python3 visualize_diagnostic.py /tmp/lio_sam_diagnostic_20240101_120000.csv")
        print("\nNote: This script will continuously read and display data from the CSV file.")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    print(f"Starting real-time visualization of: {csv_file}")
    print("Press Ctrl+C to stop")

    visualizer = DiagnosticVisualizer(csv_file)
    visualizer.animate()

if __name__ == "__main__":
    main()