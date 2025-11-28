#!/usr/bin/env python3
"""
实时轨迹绘图脚本
订阅话题并实时更新图像
"""

import rospy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2, Imu
import os
from collections import deque
from tf.transformations import euler_from_quaternion

from lio_sam.msg import cloud_info

class RealtimeTrajectoryPlotter:
    def __init__(self, output_dir='/tmp'):
        self.output_dir = output_dir

        # Data storage (use deque for memory efficiency)
        self.fusion_data = deque(maxlen=50000)
        self.gps_data = deque(maxlen=50000)
        self.degraded_enter_times = []  # Times when entering degraded zone
        self.degraded_exit_times = []   # Times when leaving degraded zone

        # Latency monitoring
        self.fusion_latency = deque(maxlen=50000)  # Store latency values
        self.gps_latency = deque(maxlen=50000)     # GPS processing latency
        self.imu_lidar_latency = deque(maxlen=50000)  # IMU-LiDAR sync latency

        # IMU data storage
        self.imu_angular_velocity = deque(maxlen=50000)  # [t, wx, wy, wz]
        self.imu_linear_acceleration = deque(maxlen=50000)  # [t, ax, ay, az]
        self.imu_orientation = deque(maxlen=50000)  # [t, roll, pitch, yaw]

        self.last_degraded_state = False
        self.start_time = None
        self.last_plot_time = 0
        self.plot_interval = 100.0  # Update plots every 5 seconds

        # Subscribe to topics
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.fusion_callback)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_callback)
        rospy.Subscriber('/gnss_degraded', Bool, self.degraded_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Timer for periodic plotting
        rospy.Timer(rospy.Duration(self.plot_interval), self.plot_callback)

        rospy.loginfo("Realtime Trajectory Plotter started")
        rospy.loginfo("Output directory: %s", output_dir)
        rospy.loginfo("Plot update interval: %.1f seconds", self.plot_interval)

    def fusion_callback(self, msg):
        t = msg.header.stamp.to_sec()
        receive_time = rospy.Time.now().to_sec()

        # Calculate processing latency (difference between now and data timestamp)
        latency = receive_time - t

        if self.start_time is None:
            self.start_time = t
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.fusion_data.append([t, x, y, z])

        # Store latency data with relative timestamp
        self.fusion_latency.append([t - self.start_time, latency * 1000])  # Convert to ms

    def gps_callback(self, msg):
        t = msg.header.stamp.to_sec()
        receive_time = rospy.Time.now().to_sec()

        # Calculate GPS processing latency
        latency = receive_time - t

        if self.start_time is None:
            self.start_time = t
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.gps_data.append([t, x, y, z])

        # Store GPS latency
        self.gps_latency.append([t - self.start_time, latency * 1000])  # Convert to ms

    def degraded_callback(self, msg):
        t = rospy.Time.now().to_sec()
        if msg.data and not self.last_degraded_state:
            self.degraded_enter_times.append(t)
            rospy.loginfo("Entering GNSS degraded zone at t=%.1f", t - (self.start_time or t))
        elif not msg.data and self.last_degraded_state:
            self.degraded_exit_times.append(t)
            rospy.loginfo("Leaving GNSS degraded zone at t=%.1f", t - (self.start_time or t))
        self.last_degraded_state = msg.data

    def imu_callback(self, msg):
        t = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = t

        rel_t = t - self.start_time

        # Angular velocity (rad/s)
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z
        self.imu_angular_velocity.append([rel_t, wx, wy, wz])

        # Linear acceleration (m/s^2)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        self.imu_linear_acceleration.append([rel_t, ax, ay, az])

        # Orientation (convert quaternion to euler angles)
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        # Check if orientation is valid (some IMUs don't provide orientation)
        if qx != 0 or qy != 0 or qz != 0 or qw != 0:
            try:
                roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
                # Convert to degrees for easier visualization
                self.imu_orientation.append([rel_t, np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])
            except:
                pass

    def plot_callback(self, event):
        if len(self.fusion_data) < 10 and len(self.gps_data) < 10:
            rospy.loginfo("Not enough data yet (fusion: %d, gps: %d)",
                         len(self.fusion_data), len(self.gps_data))
            return

        try:
            self.generate_plots()
            rospy.loginfo("Plots updated (fusion: %d, gps: %d points)",
                         len(self.fusion_data), len(self.gps_data))
        except Exception as e:
            rospy.logwarn("Failed to generate plots: %s", str(e))

    def generate_plots(self):
        # Convert to numpy arrays
        fusion = np.array(list(self.fusion_data)) if self.fusion_data else np.array([]).reshape(0, 4)
        gps = np.array(list(self.gps_data)) if self.gps_data else np.array([]).reshape(0, 4)
        fusion_lat = np.array(list(self.fusion_latency)) if self.fusion_latency else np.array([]).reshape(0, 2)
        gps_lat = np.array(list(self.gps_latency)) if self.gps_latency else np.array([]).reshape(0, 2)

        # IMU data arrays
        imu_gyro = np.array(list(self.imu_angular_velocity)) if self.imu_angular_velocity else np.array([]).reshape(0, 4)
        imu_accel = np.array(list(self.imu_linear_acceleration)) if self.imu_linear_acceleration else np.array([]).reshape(0, 4)
        imu_orient = np.array(list(self.imu_orientation)) if self.imu_orientation else np.array([]).reshape(0, 4)

        # Create figure with multiple subplots (4x4 grid now)
        fig = plt.figure(figsize=(24, 20))

        # 1. 2D Trajectory (XY)
        ax1 = fig.add_subplot(4, 4, 1)
        if len(fusion) > 0:
            ax1.plot(fusion[:, 1], fusion[:, 2], 'b-', linewidth=1, label='LIO-SAM Fusion', alpha=0.8)
        if len(gps) > 0:
            ax1.plot(gps[:, 1], gps[:, 2], 'r--', linewidth=1, label='GPS', alpha=0.6)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('2D Trajectory (Top View)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')

        # 2. 3D Trajectory
        ax2 = fig.add_subplot(4, 4, 2, projection='3d')
        if len(fusion) > 0:
            ax2.plot3D(fusion[:, 1], fusion[:, 2], fusion[:, 3], 'b-', linewidth=1, label='LIO-SAM')
        if len(gps) > 0:
            ax2.plot3D(gps[:, 1], gps[:, 2], gps[:, 3], 'r--', linewidth=1, label='GPS')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.set_title('3D Trajectory')
        ax2.legend()

        # 3. X vs Time
        ax3 = fig.add_subplot(4, 4, 3)
        if len(fusion) > 0:
            t0 = fusion[0, 0]
            ax3.plot(fusion[:, 0] - t0, fusion[:, 1], 'b-', linewidth=1, label='Fusion')
        if len(gps) > 0:
            t0 = gps[0, 0] if len(fusion) == 0 else fusion[0, 0]
            ax3.plot(gps[:, 0] - t0, gps[:, 1], 'r--', linewidth=1, label='GPS')
        # Add degraded zone markers
        t0 = fusion[0, 0] if len(fusion) > 0 else (gps[0, 0] if len(gps) > 0 else 0)
        for t in self.degraded_enter_times:
            ax3.axvline(x=t - t0, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
        for t in self.degraded_exit_times:
            ax3.axvline(x=t - t0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('X (m)')
        ax3.set_title('X Position vs Time')
        ax3.legend()
        ax3.grid(True)

        # 4. Y vs Time
        ax4 = fig.add_subplot(4, 4, 4)
        if len(fusion) > 0:
            t0 = fusion[0, 0]
            ax4.plot(fusion[:, 0] - t0, fusion[:, 2], 'b-', linewidth=1, label='Fusion')
        if len(gps) > 0:
            t0 = gps[0, 0] if len(fusion) == 0 else fusion[0, 0]
            ax4.plot(gps[:, 0] - t0, gps[:, 2], 'r--', linewidth=1, label='GPS')
        # Add degraded zone markers
        t0 = fusion[0, 0] if len(fusion) > 0 else (gps[0, 0] if len(gps) > 0 else 0)
        for t in self.degraded_enter_times:
            ax4.axvline(x=t - t0, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
        for t in self.degraded_exit_times:
            ax4.axvline(x=t - t0, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Y (m)')
        ax4.set_title('Y Position vs Time')
        ax4.legend()
        ax4.grid(True)

        # 5. Z vs Time
        ax5 = fig.add_subplot(4, 4, 5)
        if len(fusion) > 0:
            t0 = fusion[0, 0]
            ax5.plot(fusion[:, 0] - t0, fusion[:, 3], 'b-', linewidth=1, label='Fusion')
        if len(gps) > 0:
            t0 = gps[0, 0] if len(fusion) == 0 else fusion[0, 0]
            ax5.plot(gps[:, 0] - t0, gps[:, 3], 'r--', linewidth=1, label='GPS')
        # Add degraded zone markers
        t0 = fusion[0, 0] if len(fusion) > 0 else (gps[0, 0] if len(gps) > 0 else 0)
        for t in self.degraded_enter_times:
            ax5.axvline(x=t - t0, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Enter degraded' if t == self.degraded_enter_times[0] else '')
        for t in self.degraded_exit_times:
            ax5.axvline(x=t - t0, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Exit degraded' if t == self.degraded_exit_times[0] else '')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Z (m)')
        ax5.set_title('Z Position vs Time')
        ax5.legend()
        ax5.grid(True)

        # 6. Statistics
        ax6 = fig.add_subplot(4, 4, 6)
        ax6.axis('off')

        stats_text = "Statistics:\n\n"
        stats_text += f"Fusion points: {len(fusion)}\n"
        stats_text += f"GPS points: {len(gps)}\n"
        stats_text += f"IMU points: {len(imu_gyro)}\n\n"

        if len(fusion) > 1:
            duration = fusion[-1, 0] - fusion[0, 0]
            stats_text += f"Duration: {duration:.1f} s\n"

            # Calculate total distance
            diffs = np.diff(fusion[:, 1:4], axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            total_dist = np.sum(distances)
            stats_text += f"Total distance: {total_dist:.1f} m\n"
            stats_text += f"Avg speed: {total_dist/duration:.2f} m/s\n\n"

            # Position range
            stats_text += f"X range: [{fusion[:, 1].min():.1f}, {fusion[:, 1].max():.1f}] m\n"
            stats_text += f"Y range: [{fusion[:, 2].min():.1f}, {fusion[:, 2].max():.1f}] m\n"
            stats_text += f"Z range: [{fusion[:, 3].min():.1f}, {fusion[:, 3].max():.1f}] m\n"

        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')

        # 7. LIO-SAM Processing Latency
        ax7 = fig.add_subplot(4, 4, 7)
        if len(fusion_lat) > 0:
            ax7.plot(fusion_lat[:, 0], fusion_lat[:, 1], 'b-', linewidth=1, alpha=0.8)
            # Add moving average for smooth trend
            if len(fusion_lat) > 10:
                window_size = min(50, len(fusion_lat) // 10)
                moving_avg = np.convolve(fusion_lat[:, 1], np.ones(window_size)/window_size, mode='valid')
                ax7.plot(fusion_lat[window_size-1:, 0], moving_avg, 'r-', linewidth=2, alpha=0.7, label=f'Moving Avg ({window_size})')

            # Add degraded zone markers
            for t in self.degraded_enter_times:
                ax7.axvline(x=t - (self.start_time or 0), color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
            for t in self.degraded_exit_times:
                ax7.axvline(x=t - (self.start_time or 0), color='green', linestyle='--', linewidth=1.5, alpha=0.8)

        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Latency (ms)')
        ax7.set_title('LIO-SAM Processing Latency')
        ax7.grid(True, alpha=0.3)
        ax7.legend()

        # 8. GPS Processing Latency
        ax8 = fig.add_subplot(4, 4, 8)
        if len(gps_lat) > 0:
            ax8.plot(gps_lat[:, 0], gps_lat[:, 1], 'g-', linewidth=1, alpha=0.8)
            # Add moving average
            if len(gps_lat) > 10:
                window_size = min(50, len(gps_lat) // 10)
                moving_avg = np.convolve(gps_lat[:, 1], np.ones(window_size)/window_size, mode='valid')
                ax8.plot(gps_lat[window_size-1:, 0], moving_avg, 'r-', linewidth=2, alpha=0.7, label=f'Moving Avg ({window_size})')

        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Latency (ms)')
        ax8.set_title('GPS Processing Latency')
        ax8.grid(True, alpha=0.3)
        ax8.legend()

        # 9. IMU Angular Velocity (Gyroscope)
        ax9 = fig.add_subplot(4, 4, 9)
        if len(imu_gyro) > 0:
            ax9.plot(imu_gyro[:, 0], imu_gyro[:, 1], 'r-', linewidth=0.8, alpha=0.8, label='wx')
            ax9.plot(imu_gyro[:, 0], imu_gyro[:, 2], 'g-', linewidth=0.8, alpha=0.8, label='wy')
            ax9.plot(imu_gyro[:, 0], imu_gyro[:, 3], 'b-', linewidth=0.8, alpha=0.8, label='wz')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Angular Velocity (rad/s)')
        ax9.set_title('IMU Gyroscope Data')
        ax9.legend(loc='upper right')
        ax9.grid(True, alpha=0.3)

        # 10. IMU Linear Acceleration (Accelerometer)
        ax10 = fig.add_subplot(4, 4, 10)
        if len(imu_accel) > 0:
            ax10.plot(imu_accel[:, 0], imu_accel[:, 1], 'r-', linewidth=0.8, alpha=0.8, label='ax')
            ax10.plot(imu_accel[:, 0], imu_accel[:, 2], 'g-', linewidth=0.8, alpha=0.8, label='ay')
            ax10.plot(imu_accel[:, 0], imu_accel[:, 3], 'b-', linewidth=0.8, alpha=0.8, label='az')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Linear Acceleration (m/s²)')
        ax10.set_title('IMU Accelerometer Data')
        ax10.legend(loc='upper right')
        ax10.grid(True, alpha=0.3)

        # 11. IMU Orientation (Roll, Pitch, Yaw)
        ax11 = fig.add_subplot(4, 4, 11)
        if len(imu_orient) > 0:
            ax11.plot(imu_orient[:, 0], imu_orient[:, 1], 'r-', linewidth=0.8, alpha=0.8, label='Roll')
            ax11.plot(imu_orient[:, 0], imu_orient[:, 2], 'g-', linewidth=0.8, alpha=0.8, label='Pitch')
            ax11.plot(imu_orient[:, 0], imu_orient[:, 3], 'b-', linewidth=0.8, alpha=0.8, label='Yaw')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Angle (degrees)')
        ax11.set_title('IMU Orientation (Euler Angles)')
        ax11.legend(loc='upper right')
        ax11.grid(True, alpha=0.3)

        # 12. IMU Statistics
        ax12 = fig.add_subplot(4, 4, 12)
        ax12.axis('off')

        imu_stats = "IMU Statistics:\n\n"

        if len(imu_gyro) > 0:
            imu_stats += "Angular Velocity (rad/s):\n"
            imu_stats += f"  wx: mean={np.mean(imu_gyro[:, 1]):.4f}, std={np.std(imu_gyro[:, 1]):.4f}\n"
            imu_stats += f"  wy: mean={np.mean(imu_gyro[:, 2]):.4f}, std={np.std(imu_gyro[:, 2]):.4f}\n"
            imu_stats += f"  wz: mean={np.mean(imu_gyro[:, 3]):.4f}, std={np.std(imu_gyro[:, 3]):.4f}\n\n"

        if len(imu_accel) > 0:
            imu_stats += "Linear Accel (m/s²):\n"
            imu_stats += f"  ax: mean={np.mean(imu_accel[:, 1]):.4f}, std={np.std(imu_accel[:, 1]):.4f}\n"
            imu_stats += f"  ay: mean={np.mean(imu_accel[:, 2]):.4f}, std={np.std(imu_accel[:, 2]):.4f}\n"
            imu_stats += f"  az: mean={np.mean(imu_accel[:, 3]):.4f}, std={np.std(imu_accel[:, 3]):.4f}\n\n"

        if len(imu_orient) > 0:
            imu_stats += "Orientation (deg):\n"
            imu_stats += f"  Roll:  [{np.min(imu_orient[:, 1]):.1f}, {np.max(imu_orient[:, 1]):.1f}]\n"
            imu_stats += f"  Pitch: [{np.min(imu_orient[:, 2]):.1f}, {np.max(imu_orient[:, 2]):.1f}]\n"
            imu_stats += f"  Yaw:   [{np.min(imu_orient[:, 3]):.1f}, {np.max(imu_orient[:, 3]):.1f}]\n"

        ax12.text(0.1, 0.5, imu_stats, transform=ax12.transAxes, fontsize=9,
                 verticalalignment='center', fontfamily='monospace')

        # 13. IMU Gyroscope Norm (Motion Intensity)
        ax13 = fig.add_subplot(4, 4, 13)
        if len(imu_gyro) > 0:
            gyro_norm = np.sqrt(imu_gyro[:, 1]**2 + imu_gyro[:, 2]**2 + imu_gyro[:, 3]**2)
            ax13.plot(imu_gyro[:, 0], gyro_norm, 'm-', linewidth=0.8, alpha=0.8)
            # Add moving average
            if len(gyro_norm) > 50:
                window_size = min(100, len(gyro_norm) // 10)
                moving_avg = np.convolve(gyro_norm, np.ones(window_size)/window_size, mode='valid')
                ax13.plot(imu_gyro[window_size-1:, 0], moving_avg, 'k-', linewidth=1.5, alpha=0.7, label=f'Avg ({window_size})')
        ax13.set_xlabel('Time (s)')
        ax13.set_ylabel('Gyro Norm (rad/s)')
        ax13.set_title('Rotation Intensity')
        ax13.legend(loc='upper right')
        ax13.grid(True, alpha=0.3)

        # 14. IMU Accelerometer Norm (Gravity + Dynamic)
        ax14 = fig.add_subplot(4, 4, 14)
        if len(imu_accel) > 0:
            accel_norm = np.sqrt(imu_accel[:, 1]**2 + imu_accel[:, 2]**2 + imu_accel[:, 3]**2)
            ax14.plot(imu_accel[:, 0], accel_norm, 'c-', linewidth=0.8, alpha=0.8)
            ax14.axhline(y=9.81, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Gravity (9.81)')
            # Add moving average
            if len(accel_norm) > 50:
                window_size = min(100, len(accel_norm) // 10)
                moving_avg = np.convolve(accel_norm, np.ones(window_size)/window_size, mode='valid')
                ax14.plot(imu_accel[window_size-1:, 0], moving_avg, 'k-', linewidth=1.5, alpha=0.7, label=f'Avg ({window_size})')
        ax14.set_xlabel('Time (s)')
        ax14.set_ylabel('Accel Norm (m/s²)')
        ax14.set_title('Acceleration Magnitude')
        ax14.legend(loc='upper right')
        ax14.grid(True, alpha=0.3)

        # 15. Latency Statistics
        ax15 = fig.add_subplot(4, 4, 15)
        ax15.axis('off')

        latency_stats = "Latency Statistics:\n\n"

        if len(fusion_lat) > 0:
            latency_stats += "LIO-SAM Latency:\n"
            latency_stats += f"  Current: {fusion_lat[-1][1]:.2f} ms\n"
            latency_stats += f"  Mean: {np.mean(fusion_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  Std: {np.std(fusion_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  Min: {np.min(fusion_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  Max: {np.max(fusion_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  95%: {np.percentile(fusion_lat[:, 1], 95):.2f} ms\n\n"

        if len(gps_lat) > 0:
            latency_stats += "GPS Latency:\n"
            latency_stats += f"  Current: {gps_lat[-1][1]:.2f} ms\n"
            latency_stats += f"  Mean: {np.mean(gps_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  Std: {np.std(gps_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  Min: {np.min(gps_lat[:, 1]):.2f} ms\n"
            latency_stats += f"  Max: {np.max(gps_lat[:, 1]):.2f} ms\n"

        # Performance indicators
        if len(fusion_lat) > 0:
            mean_latency = np.mean(fusion_lat[:, 1])
            if mean_latency < 50:
                latency_stats += "\nReal-time OK"
            elif mean_latency < 100:
                latency_stats += "\nNear real-time"
            else:
                latency_stats += "\nHigh latency!"

        ax15.text(0.1, 0.5, latency_stats, transform=ax15.transAxes, fontsize=9,
                 verticalalignment='center', fontfamily='monospace')

        # 16. IMU Data Rate Monitor
        ax16 = fig.add_subplot(4, 4, 16)
        ax16.axis('off')

        rate_stats = "Data Rate Info:\n\n"

        if len(imu_gyro) > 1:
            # Calculate IMU data rate
            imu_dt = np.diff(imu_gyro[:, 0])
            imu_rate = 1.0 / np.mean(imu_dt) if np.mean(imu_dt) > 0 else 0
            rate_stats += f"IMU Rate: {imu_rate:.1f} Hz\n"
            rate_stats += f"IMU dt mean: {np.mean(imu_dt)*1000:.2f} ms\n"
            rate_stats += f"IMU dt std: {np.std(imu_dt)*1000:.2f} ms\n\n"

        if len(fusion) > 1:
            fusion_dt = np.diff(fusion[:, 0])
            fusion_rate = 1.0 / np.mean(fusion_dt) if np.mean(fusion_dt) > 0 else 0
            rate_stats += f"Fusion Rate: {fusion_rate:.1f} Hz\n"
            rate_stats += f"Fusion dt mean: {np.mean(fusion_dt)*1000:.2f} ms\n\n"

        if len(gps) > 1:
            gps_dt = np.diff(gps[:, 0])
            gps_rate = 1.0 / np.mean(gps_dt) if np.mean(gps_dt) > 0 else 0
            rate_stats += f"GPS Rate: {gps_rate:.1f} Hz\n"

        ax16.text(0.1, 0.5, rate_stats, transform=ax16.transAxes, fontsize=9,
                 verticalalignment='center', fontfamily='monospace')

        plt.tight_layout()

        # Save plot
        output_path = os.path.join(self.output_dir, 'trajectory_realtime.png')
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    def save_final_data(self):
        """Save final data to CSV files"""
        if len(self.fusion_data) > 0:
            fusion = np.array(list(self.fusion_data))
            np.savetxt(os.path.join(self.output_dir, 'fusion_trajectory.csv'),
                      fusion, delimiter=',', header='time,x,y,z', comments='')
            rospy.loginfo("Saved fusion trajectory: %d points", len(fusion))

        if len(self.gps_data) > 0:
            gps = np.array(list(self.gps_data))
            np.savetxt(os.path.join(self.output_dir, 'gps_trajectory.csv'),
                      gps, delimiter=',', header='time,x,y,z', comments='')
            rospy.loginfo("Saved GPS trajectory: %d points", len(gps))

        # Save latency data
        if len(self.fusion_latency) > 0:
            fusion_lat = np.array(list(self.fusion_latency))
            np.savetxt(os.path.join(self.output_dir, 'fusion_latency.csv'),
                      fusion_lat, delimiter=',', header='relative_time_s,latency_ms', comments='')
            rospy.loginfo("Saved fusion latency: %d points, mean=%.2f ms",
                         len(fusion_lat), np.mean(fusion_lat[:, 1]))

        if len(self.gps_latency) > 0:
            gps_lat = np.array(list(self.gps_latency))
            np.savetxt(os.path.join(self.output_dir, 'gps_latency.csv'),
                      gps_lat, delimiter=',', header='relative_time_s,latency_ms', comments='')
            rospy.loginfo("Saved GPS latency: %d points, mean=%.2f ms",
                         len(gps_lat), np.mean(gps_lat[:, 1]))

        # Save IMU data
        if len(self.imu_angular_velocity) > 0:
            imu_gyro = np.array(list(self.imu_angular_velocity))
            np.savetxt(os.path.join(self.output_dir, 'imu_angular_velocity.csv'),
                      imu_gyro, delimiter=',', header='relative_time_s,wx_rad_s,wy_rad_s,wz_rad_s', comments='')
            rospy.loginfo("Saved IMU angular velocity: %d points", len(imu_gyro))

        if len(self.imu_linear_acceleration) > 0:
            imu_accel = np.array(list(self.imu_linear_acceleration))
            np.savetxt(os.path.join(self.output_dir, 'imu_linear_acceleration.csv'),
                      imu_accel, delimiter=',', header='relative_time_s,ax_m_s2,ay_m_s2,az_m_s2', comments='')
            rospy.loginfo("Saved IMU linear acceleration: %d points", len(imu_accel))

        if len(self.imu_orientation) > 0:
            imu_orient = np.array(list(self.imu_orientation))
            np.savetxt(os.path.join(self.output_dir, 'imu_orientation.csv'),
                      imu_orient, delimiter=',', header='relative_time_s,roll_deg,pitch_deg,yaw_deg', comments='')
            rospy.loginfo("Saved IMU orientation: %d points", len(imu_orient))

if __name__ == '__main__':
    rospy.init_node('realtime_trajectory_plotter')

    output_dir = rospy.get_param('~output_dir', './output')

    plotter = RealtimeTrajectoryPlotter(output_dir)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Generate final plot and save data
        rospy.loginfo("Generating final plots...")
        plotter.generate_plots()
        plotter.save_final_data()
        rospy.loginfo("Done. Output saved to %s", output_dir)
