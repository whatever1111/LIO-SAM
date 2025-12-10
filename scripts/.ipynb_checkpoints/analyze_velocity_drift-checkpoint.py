#!/usr/bin/env python3
"""
Analyze velocity drift mechanism in LIO-SAM
This script monitors:
1. Degenerate scan detections
2. IMU preintegration velocity
3. Optimizer output velocity
4. Velocity drift correlation with degenerate scans
"""

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from collections import deque
import time

class VelocityDriftAnalyzer:
    def __init__(self):
        rospy.init_node('velocity_drift_analyzer', anonymous=True)

        # Data storage
        self.imu_odom_history = deque(maxlen=1000)
        self.lidar_odom_history = deque(maxlen=100)
        self.degenerate_count = 0
        self.normal_count = 0

        self.last_lidar_time = None
        self.last_imu_vel = None
        self.velocity_jump_events = []

        # Subscribe
        rospy.Subscriber("/odometry/imu_incremental", Odometry, self.imu_odom_callback)
        rospy.Subscriber("/lio_sam/mapping/odometry_incremental", Odometry, self.lidar_odom_callback)

        print("=" * 70)
        print("Velocity Drift Analyzer - Monitoring correlation between")
        print("degenerate scans and velocity drift")
        print("=" * 70)
        print()
        print("Columns:")
        print("  Time: ROS timestamp")
        print("  IMU_Vel: Velocity from IMU preintegration (m/s)")
        print("  LiDAR_Vel: Velocity from optimizer (m/s)")
        print("  Degen: Whether last scan was degenerate")
        print("  Drift: Velocity difference between IMU and LiDAR")
        print()
        print("-" * 70)

    def imu_odom_callback(self, msg):
        t = msg.header.stamp.to_sec()
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)

        self.imu_odom_history.append({
            'time': t,
            'vel': np.array([vx, vy, vz]),
            'vel_mag': vel_mag
        })

        # Detect sudden velocity jumps
        if self.last_imu_vel is not None:
            vel_change = vel_mag - self.last_imu_vel
            if abs(vel_change) > 5.0:  # 5 m/s jump in single IMU cycle
                self.velocity_jump_events.append({
                    'time': t,
                    'from': self.last_imu_vel,
                    'to': vel_mag,
                    'change': vel_change
                })
                print(f"\n*** VELOCITY JUMP DETECTED ***")
                print(f"    Time: {t:.3f}")
                print(f"    From {self.last_imu_vel:.2f} to {vel_mag:.2f} m/s")
                print(f"    Change: {vel_change:+.2f} m/s")
                print()

        self.last_imu_vel = vel_mag

    def lidar_odom_callback(self, msg):
        t = msg.header.stamp.to_sec()

        # Check if degenerate (stored in covariance[0])
        degenerate = int(msg.pose.covariance[0]) == 1

        # Get velocity from LiDAR odometry (if available)
        # Note: mapping/odometry_incremental doesn't have twist, we need to compute from pose change
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z

        lidar_vel_mag = 0
        if self.last_lidar_time is not None and len(self.lidar_odom_history) > 0:
            last = self.lidar_odom_history[-1]
            dt = t - last['time']
            if dt > 0.01:
                dx = px - last['pos'][0]
                dy = py - last['pos'][1]
                dz = pz - last['pos'][2]
                lidar_vel_mag = np.sqrt(dx**2 + dy**2 + dz**2) / dt

        self.lidar_odom_history.append({
            'time': t,
            'pos': np.array([px, py, pz]),
            'degenerate': degenerate,
            'vel_mag': lidar_vel_mag
        })

        if degenerate:
            self.degenerate_count += 1
        else:
            self.normal_count += 1

        # Find closest IMU odometry
        imu_vel_mag = 0
        imu_vel = np.zeros(3)
        if len(self.imu_odom_history) > 0:
            # Find IMU data closest to this time
            min_dt = float('inf')
            for imu in self.imu_odom_history:
                dt = abs(imu['time'] - t)
                if dt < min_dt:
                    min_dt = dt
                    imu_vel_mag = imu['vel_mag']
                    imu_vel = imu['vel']

        drift = imu_vel_mag - lidar_vel_mag

        total = self.degenerate_count + self.normal_count
        degen_pct = (self.degenerate_count / total * 100) if total > 0 else 0

        status = "DEGEN" if degenerate else "OK"
        print(f"t={t:.2f} IMU:{imu_vel_mag:6.2f}m/s LiDAR:{lidar_vel_mag:6.2f}m/s [{status:5s}] "
              f"Drift:{drift:+6.2f}m/s Degen:{degen_pct:5.1f}%")

        # Print detailed velocity components if drift is large
        if abs(drift) > 2.0:
            print(f"         IMU vel components: x={imu_vel[0]:+.2f} y={imu_vel[1]:+.2f} z={imu_vel[2]:+.2f}")

        self.last_lidar_time = t

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        analyzer = VelocityDriftAnalyzer()
        analyzer.run()
    except rospy.ROSInterruptException:
        pass
