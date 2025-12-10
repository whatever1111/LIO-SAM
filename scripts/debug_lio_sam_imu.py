#!/usr/bin/env python3
"""
Debug script to verify lio_sam/imu/data orientation vs /imu/data raw.
Run this while LIO-SAM is running.
"""

import rospy
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation as R
import numpy as np

class ImuDebugger:
    def __init__(self):
        self.raw_imu_ori = None
        self.lio_sam_imu_ori = None
        self.count = 0

        # Expected extrinsicRPY = Rz(90)
        # extQRPY = inverse of extrinsicRPY = Rz(-90)
        angle = np.radians(90)  # Rz(90)
        self.extRPY = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        self.extQRPY = R.from_matrix(self.extRPY).inv()

    def raw_imu_callback(self, msg):
        self.raw_imu_ori = msg.orientation

    def lio_sam_imu_callback(self, msg):
        self.lio_sam_imu_ori = msg.orientation
        self.count += 1

        # Print every 200 messages (~1s at 200Hz)
        if self.count % 200 == 1:
            self.debug_output()

    def debug_output(self):
        if self.raw_imu_ori is None:
            print("[WARNING] No /imu/data received yet")
            return
        if self.lio_sam_imu_ori is None:
            print("[WARNING] No lio_sam/imu/data received yet")
            return

        # Raw /imu/data orientation
        q_raw = [self.raw_imu_ori.x, self.raw_imu_ori.y, self.raw_imu_ori.z, self.raw_imu_ori.w]
        r_raw = R.from_quat(q_raw)
        rpy_raw = r_raw.as_euler('xyz', degrees=True)

        # lio_sam/imu/data orientation
        q_lio = [self.lio_sam_imu_ori.x, self.lio_sam_imu_ori.y, self.lio_sam_imu_ori.z, self.lio_sam_imu_ori.w]
        r_lio = R.from_quat(q_lio)
        rpy_lio = r_lio.as_euler('xyz', degrees=True)

        # Expected: q_raw * extQRPY (extQRPY = Rz(-90))
        r_expected = r_raw * self.extQRPY
        rpy_expected = r_expected.as_euler('xyz', degrees=True)

        print("\n" + "="*60)
        print(f"IMU Orientation Debug (msg #{self.count})")
        print("="*60)
        print(f"/imu/data raw:        Roll={rpy_raw[0]:+7.2f}, Pitch={rpy_raw[1]:+7.2f}, Yaw={rpy_raw[2]:+7.2f}")
        print(f"Expected (q*extQRPY): Roll={rpy_expected[0]:+7.2f}, Pitch={rpy_expected[1]:+7.2f}, Yaw={rpy_expected[2]:+7.2f}")
        print(f"lio_sam/imu/data:     Roll={rpy_lio[0]:+7.2f}, Pitch={rpy_lio[1]:+7.2f}, Yaw={rpy_lio[2]:+7.2f}")

        yaw_diff = rpy_lio[2] - rpy_expected[2]
        while yaw_diff > 180: yaw_diff -= 360
        while yaw_diff < -180: yaw_diff += 360

        if abs(yaw_diff) < 5:
            print(f"[OK] Yaw matches expected (diff: {yaw_diff:+.2f} deg)")
        else:
            print(f"[ERROR] Yaw MISMATCH! diff = {yaw_diff:+.2f} deg")
            print(f"        Expected yaw change from Rz(-90): {rpy_raw[2]:+.2f} -> {rpy_expected[2]:+.2f}")
            print(f"        Actual lio_sam/imu/data yaw:      {rpy_lio[2]:+.2f}")


def main():
    rospy.init_node('imu_orientation_debugger', anonymous=True)

    debugger = ImuDebugger()

    rospy.Subscriber('/imu/data', Imu, debugger.raw_imu_callback)
    rospy.Subscriber('lio_sam/imu/data', Imu, debugger.lio_sam_imu_callback)

    print("="*60)
    print("IMU Orientation Debugger")
    print("="*60)
    print("Subscribed to:")
    print("  - /imu/data (raw)")
    print("  - lio_sam/imu/data (transformed)")
    print("")
    print(f"Expected transform: q * extQRPY where extQRPY = Rz(-90)")
    print(f"This should subtract 90 deg from yaw")
    print("="*60)

    rospy.spin()


if __name__ == "__main__":
    main()
