#!/usr/bin/env python3
"""
检查FPA IMU的Z轴方向
静止状态下，重力加速度应该在某个轴上约为9.8 m/s^2
"""
import rospy
import numpy as np
from collections import deque

try:
    from fixposition_driver_msgs.msg import FpaImu
    HAS_FPAIMU = True
except ImportError:
    HAS_FPAIMU = False
    print("ERROR: fixposition_driver_msgs not found")

class FpaImuChecker:
    def __init__(self):
        rospy.init_node('fpa_imu_checker', anonymous=True)
        self.acc_data = deque(maxlen=200)
        self.gyr_data = deque(maxlen=200)

        if HAS_FPAIMU:
            rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.callback)

        rospy.Timer(rospy.Duration(3.0), self.analyze)
        print("Waiting for FPA IMU data...")

    def callback(self, msg):
        acc = np.array([msg.data.linear_acceleration.x,
                       msg.data.linear_acceleration.y,
                       msg.data.linear_acceleration.z])
        gyr = np.array([msg.data.angular_velocity.x,
                       msg.data.angular_velocity.y,
                       msg.data.angular_velocity.z])
        self.acc_data.append(acc)
        self.gyr_data.append(gyr)

    def analyze(self, event):
        if len(self.acc_data) < 50:
            print(f"Collecting data... ({len(self.acc_data)}/50)")
            return

        acc_mean = np.mean(self.acc_data, axis=0)
        acc_std = np.std(self.acc_data, axis=0)
        gyr_mean = np.mean(self.gyr_data, axis=0)

        print("\n" + "="*60)
        print("FPA IMU Raw Data Analysis (CORRIMU)")
        print("="*60)
        print(f"\nAcceleration (m/s^2):")
        print(f"  X: {acc_mean[0]:+8.4f} +/- {acc_std[0]:.4f}")
        print(f"  Y: {acc_mean[1]:+8.4f} +/- {acc_std[1]:.4f}")
        print(f"  Z: {acc_mean[2]:+8.4f} +/- {acc_std[2]:.4f}")
        print(f"  |acc|: {np.linalg.norm(acc_mean):.4f}")

        print(f"\nAngular Velocity (rad/s):")
        print(f"  X: {gyr_mean[0]:+8.6f}")
        print(f"  Y: {gyr_mean[1]:+8.6f}")
        print(f"  Z: {gyr_mean[2]:+8.6f}")

        # 判断重力方向
        print("\n" + "-"*60)
        print("Gravity Direction Analysis:")
        gravity_axis = np.argmax(np.abs(acc_mean))
        axis_names = ['X', 'Y', 'Z']
        gravity_sign = '+' if acc_mean[gravity_axis] > 0 else '-'

        print(f"  Gravity is mainly in {gravity_sign}{axis_names[gravity_axis]} axis")
        print(f"  Value: {acc_mean[gravity_axis]:+.4f} m/s^2")

        # LIO-SAM期望
        print("\n" + "-"*60)
        print("LIO-SAM Expectation:")
        print("  LIO-SAM expects gravity in -Z direction (acc_z ~ -9.8)")

        # 当前extrinsicRot效果
        ext_rot_180 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])  # Rz(180)
        ext_rot_flip_z = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])  # Rz(180) + flip Z

        acc_after_180 = ext_rot_180 @ acc_mean
        acc_after_flip = ext_rot_flip_z @ acc_mean

        print("\n" + "-"*60)
        print("After extrinsicRot transformation:")
        print(f"\n  Current Rz(180): [-1,0,0, 0,-1,0, 0,0,1]")
        print(f"    Z: {acc_after_180[2]:+.4f} m/s^2 {'OK' if acc_after_180[2] < -8 else 'WRONG!'}")

        print(f"\n  With Z-flip: [-1,0,0, 0,-1,0, 0,0,-1]")
        print(f"    Z: {acc_after_flip[2]:+.4f} m/s^2 {'OK' if acc_after_flip[2] < -8 else 'WRONG!'}")

        # 建议
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        if acc_after_180[2] < -8:
            print("  Current extrinsicRot is CORRECT")
            print("  extrinsicRot: [-1, 0, 0, 0, -1, 0, 0, 0, 1]")
        elif acc_after_flip[2] < -8:
            print("  Need to FLIP Z axis!")
            print("  extrinsicRot: [-1, 0, 0, 0, -1, 0, 0, 0, -1]")
        else:
            print("  Neither configuration works. Check IMU mounting.")
        print("="*60)

if __name__ == '__main__':
    checker = FpaImuChecker()
    rospy.spin()
