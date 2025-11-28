#!/usr/bin/env python3
"""
LIO-SAM Runtime Monitor
=======================
Real-time monitoring of LIO-SAM state during operation.
Detects velocity anomalies, coordinate issues, and other problems.

Usage:
    rosrun lio_sam runtime_monitor.py
"""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class RuntimeMonitor:
    def __init__(self):
        rospy.init_node('runtime_monitor', anonymous=True)

        # Data storage
        self.imu_data = []
        self.gps_data = []
        self.lio_odom_data = []
        self.imu_preint_data = []

        # State
        self.last_print_time = 0
        self.anomaly_count = 0
        self.velocity_history = []

        # Thresholds
        self.velocity_threshold = 30.0  # m/s
        self.acceleration_threshold = 10.0  # m/s^2
        self.angular_velocity_threshold = 3.0  # rad/s

        # Locks
        self.lock = threading.Lock()

        # Subscribers
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.lio_odom_callback)
        rospy.Subscriber('/odometry/imu_incremental', Odometry, self.imu_preint_callback)

        # Timer for periodic status
        rospy.Timer(rospy.Duration(1.0), self.print_status)

        print(f"{Colors.BOLD}LIO-SAM Runtime Monitor Started{Colors.RESET}")
        print("="*60)
        print("Monitoring for velocity anomalies and coordinate issues...")
        print()

    def imu_callback(self, msg):
        with self.lock:
            acc = np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z])
            gyro = np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])

            # Check for anomalies
            acc_norm = np.linalg.norm(acc)
            gyro_norm = np.linalg.norm(gyro)

            if acc_norm > 50:  # Way too high
                self.log_anomaly(f"IMU ACC SPIKE: {acc_norm:.2f} m/s^2", msg.header.stamp)

            if gyro_norm > self.angular_velocity_threshold:
                self.log_anomaly(f"IMU GYRO HIGH: {gyro_norm:.2f} rad/s", msg.header.stamp)

            self.imu_data.append({
                'time': msg.header.stamp.to_sec(),
                'acc': acc,
                'gyro': gyro,
                'quat': [msg.orientation.x, msg.orientation.y,
                        msg.orientation.z, msg.orientation.w]
            })

            # Keep only last 1000 samples
            if len(self.imu_data) > 1000:
                self.imu_data.pop(0)

    def gps_callback(self, msg):
        with self.lock:
            pos = np.array([msg.pose.pose.position.x,
                           msg.pose.pose.position.y,
                           msg.pose.pose.position.z])

            # Calculate velocity from position if we have previous data
            if len(self.gps_data) > 0:
                prev = self.gps_data[-1]
                dt = msg.header.stamp.to_sec() - prev['time']
                if dt > 0:
                    vel = (pos - prev['pos']) / dt
                    speed = np.linalg.norm(vel)

                    if speed > self.velocity_threshold:
                        self.log_anomaly(f"GPS VELOCITY SPIKE: {speed:.2f} m/s", msg.header.stamp)

            self.gps_data.append({
                'time': msg.header.stamp.to_sec(),
                'pos': pos,
                'frame': msg.header.frame_id
            })

            if len(self.gps_data) > 500:
                self.gps_data.pop(0)

    def lio_odom_callback(self, msg):
        with self.lock:
            pos = np.array([msg.pose.pose.position.x,
                           msg.pose.pose.position.y,
                           msg.pose.pose.position.z])

            quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                   msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

            # Get velocity from twist
            vel = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.linear.y,
                           msg.twist.twist.linear.z])
            speed = np.linalg.norm(vel)

            self.velocity_history.append(speed)
            if len(self.velocity_history) > 100:
                self.velocity_history.pop(0)

            if speed > self.velocity_threshold:
                self.log_anomaly(f"LIO VELOCITY SPIKE: {speed:.2f} m/s, pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]",
                               msg.header.stamp)

            # Calculate velocity from position change
            if len(self.lio_odom_data) > 0:
                prev = self.lio_odom_data[-1]
                dt = msg.header.stamp.to_sec() - prev['time']
                if dt > 0:
                    pos_vel = (pos - prev['pos']) / dt
                    pos_speed = np.linalg.norm(pos_vel)

                    if pos_speed > self.velocity_threshold:
                        self.log_anomaly(f"LIO POS-DERIVED VEL SPIKE: {pos_speed:.2f} m/s", msg.header.stamp)

            self.lio_odom_data.append({
                'time': msg.header.stamp.to_sec(),
                'pos': pos,
                'quat': quat,
                'vel': vel
            })

            if len(self.lio_odom_data) > 500:
                self.lio_odom_data.pop(0)

    def imu_preint_callback(self, msg):
        with self.lock:
            vel = np.array([msg.twist.twist.linear.x,
                           msg.twist.twist.linear.y,
                           msg.twist.twist.linear.z])
            speed = np.linalg.norm(vel)

            if speed > self.velocity_threshold:
                self.log_anomaly(f"IMU_PREINT VELOCITY SPIKE: {speed:.2f} m/s", msg.header.stamp)

            self.imu_preint_data.append({
                'time': msg.header.stamp.to_sec(),
                'vel': vel,
                'speed': speed
            })

            if len(self.imu_preint_data) > 500:
                self.imu_preint_data.pop(0)

    def log_anomaly(self, message, stamp):
        self.anomaly_count += 1
        t = stamp.to_sec()
        print(f"{Colors.RED}[ANOMALY #{self.anomaly_count}] t={t:.3f}: {message}{Colors.RESET}")

    def print_status(self, event):
        with self.lock:
            print("\n" + "="*60)
            print(f"{Colors.BOLD}Runtime Status @ {rospy.Time.now().to_sec():.3f}{Colors.RESET}")
            print("="*60)

            # IMU status
            if len(self.imu_data) > 0:
                recent_imu = self.imu_data[-10:]
                acc_mean = np.mean([d['acc'] for d in recent_imu], axis=0)
                gyro_mean = np.mean([d['gyro'] for d in recent_imu], axis=0)

                # Get yaw from quaternion
                quat = self.imu_data[-1]['quat']
                r = R.from_quat(quat)
                euler = r.as_euler('xyz', degrees=True)

                print(f"\n{Colors.BLUE}IMU:{Colors.RESET}")
                print(f"  Samples: {len(self.imu_data)}")
                print(f"  Acc mean: [{acc_mean[0]:.3f}, {acc_mean[1]:.3f}, {acc_mean[2]:.3f}] m/s^2")
                print(f"  Gyro mean: [{gyro_mean[0]:.4f}, {gyro_mean[1]:.4f}, {gyro_mean[2]:.4f}] rad/s")
                print(f"  Current RPY: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")

            # GPS status
            if len(self.gps_data) > 1:
                recent_gps = self.gps_data[-10:]
                pos = recent_gps[-1]['pos']

                # Calculate GPS velocity
                dt = recent_gps[-1]['time'] - recent_gps[0]['time']
                if dt > 0:
                    dp = recent_gps[-1]['pos'] - recent_gps[0]['pos']
                    gps_vel = np.linalg.norm(dp) / dt
                else:
                    gps_vel = 0

                print(f"\n{Colors.BLUE}GPS:{Colors.RESET}")
                print(f"  Samples: {len(self.gps_data)}")
                print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                print(f"  Avg velocity: {gps_vel:.2f} m/s")

            # LIO-SAM status
            if len(self.lio_odom_data) > 0:
                latest = self.lio_odom_data[-1]
                pos = latest['pos']
                vel = latest['vel']
                speed = np.linalg.norm(vel)

                # Get yaw
                r = R.from_quat(latest['quat'])
                euler = r.as_euler('xyz', degrees=True)

                print(f"\n{Colors.BLUE}LIO-SAM Odometry:{Colors.RESET}")
                print(f"  Samples: {len(self.lio_odom_data)}")
                print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                print(f"  Velocity: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] ({speed:.2f} m/s)")
                print(f"  RPY: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")

                if speed > 5:
                    print(f"  {Colors.YELLOW}WARNING: High velocity!{Colors.RESET}")

            # IMU Preintegration status
            if len(self.imu_preint_data) > 0:
                latest = self.imu_preint_data[-1]
                vel = latest['vel']
                speed = latest['speed']

                # Calculate average speed
                avg_speed = np.mean([d['speed'] for d in self.imu_preint_data[-20:]])
                max_speed = np.max([d['speed'] for d in self.imu_preint_data])

                print(f"\n{Colors.BLUE}IMU Preintegration:{Colors.RESET}")
                print(f"  Samples: {len(self.imu_preint_data)}")
                print(f"  Current velocity: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] ({speed:.2f} m/s)")
                print(f"  Avg speed (last 20): {avg_speed:.2f} m/s")
                print(f"  Max speed seen: {max_speed:.2f} m/s")

                if max_speed > self.velocity_threshold:
                    print(f"  {Colors.RED}ERROR: Velocity exceeded threshold!{Colors.RESET}")

            # Coordinate frame comparison
            if len(self.imu_data) > 0 and len(self.lio_odom_data) > 0:
                imu_quat = self.imu_data[-1]['quat']
                lio_quat = self.lio_odom_data[-1]['quat']

                imu_euler = R.from_quat(imu_quat).as_euler('xyz', degrees=True)
                lio_euler = R.from_quat(lio_quat).as_euler('xyz', degrees=True)

                yaw_diff = imu_euler[2] - lio_euler[2]
                while yaw_diff > 180: yaw_diff -= 360
                while yaw_diff < -180: yaw_diff += 360

                print(f"\n{Colors.BLUE}Coordinate Frame Comparison:{Colors.RESET}")
                print(f"  IMU yaw: {imu_euler[2]:.2f} deg")
                print(f"  LIO yaw: {lio_euler[2]:.2f} deg")
                print(f"  Difference: {yaw_diff:.2f} deg")

                if abs(yaw_diff) > 30:
                    print(f"  {Colors.YELLOW}WARNING: Significant yaw difference!{Colors.RESET}")

            # Anomaly summary
            print(f"\n{Colors.BOLD}Total anomalies detected: {self.anomaly_count}{Colors.RESET}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        monitor = RuntimeMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
