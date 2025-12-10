#!/usr/bin/env python3
"""
Live debug: Subscribe to different IMU topics and compare orientations.
Run this while LIO-SAM is running.
"""

import rospy
from sensor_msgs.msg import Imu
import numpy as np
from scipy.spatial.transform import Rotation as R

class ImuComparator:
    def __init__(self):
        self.imu_data_ori = None
        self.lio_sam_imu_ori = None
        self.count = 0

        # Extrinsic transform Rz(90)
        self.extRPY = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])

    def imu_data_callback(self, msg):
        """Callback for /imu/data"""
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.imu_data_ori = q

    def lio_sam_imu_callback(self, msg):
        """Callback for lio_sam/imu/data"""
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        self.lio_sam_imu_ori = q
        self.count += 1

        if self.count % 100 == 0:  # Print every 100 messages (~0.5s)
            self.compare()

    def compare(self):
        if self.imu_data_ori is None or self.lio_sam_imu_ori is None:
            return

        # /imu/data orientation
        r_imu = R.from_quat(self.imu_data_ori)
        rpy_imu = r_imu.as_euler('xyz', degrees=True)

        # lio_sam/imu/data orientation
        r_lio = R.from_quat(self.lio_sam_imu_ori)
        rpy_lio = r_lio.as_euler('xyz', degrees=True)

        # Expected after transform: /imu/data * extQRPY^(-1)
        r_extRPY = R.from_matrix(self.extRPY)
        extQRPY = r_extRPY.inv()
        r_expected = r_imu * extQRPY
        rpy_expected = r_expected.as_euler('xyz', degrees=True)

        print("\n" + "="*60)
        print(f"IMU Orientation Comparison (count={self.count})")
        print("="*60)
        print(f"/imu/data raw:        Roll={rpy_imu[0]:+7.2f}, Pitch={rpy_imu[1]:+7.2f}, Yaw={rpy_imu[2]:+7.2f}")
        print(f"Expected (q*extQRPY): Roll={rpy_expected[0]:+7.2f}, Pitch={rpy_expected[1]:+7.2f}, Yaw={rpy_expected[2]:+7.2f}")
        print(f"lio_sam/imu/data:     Roll={rpy_lio[0]:+7.2f}, Pitch={rpy_lio[1]:+7.2f}, Yaw={rpy_lio[2]:+7.2f}")

        diff_yaw = rpy_lio[2] - rpy_expected[2]
        while diff_yaw > 180: diff_yaw -= 360
        while diff_yaw < -180: diff_yaw += 360

        if abs(diff_yaw) < 5:
            print(f"✓ lio_sam/imu/data matches expected (yaw diff: {diff_yaw:+.2f}°)")
        else:
            print(f"✗ MISMATCH! Yaw difference: {diff_yaw:+.2f}°")


def main():
    rospy.init_node('imu_comparator', anonymous=True)

    comp = ImuComparator()

    rospy.Subscriber('/imu/data', Imu, comp.imu_data_callback)
    rospy.Subscriber('lio_sam/imu/data', Imu, comp.lio_sam_imu_callback)

    print("Subscribing to /imu/data and lio_sam/imu/data...")
    print("Make sure LIO-SAM is running with useFpaImu=true")

    rospy.spin()


if __name__ == "__main__":
    main()
