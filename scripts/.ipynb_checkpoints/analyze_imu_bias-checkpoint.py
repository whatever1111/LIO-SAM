#!/usr/bin/env python3
"""
Analyze why CORRIMU and /imu/data have different values.
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R

BAG_FILE = "/root/autodl-tmp/info_fixed.bag"

def main():
    bag = rosbag.Bag(BAG_FILE)

    # Collect first few seconds of data when vehicle is stationary
    corrimu_data = []  # (time, acc, gyro)
    imu_data = []      # (time, acc, gyro)

    print("Reading first 5 seconds (vehicle should be stationary)...")

    start_time = None
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/corrimu', '/imu/data']):
        if start_time is None:
            start_time = t.to_sec()

        if t.to_sec() - start_time > 5:
            break

        if topic == '/fixposition/fpa/corrimu':
            ts = msg.data.header.stamp.to_sec()
            acc = np.array([msg.data.linear_acceleration.x,
                           msg.data.linear_acceleration.y,
                           msg.data.linear_acceleration.z])
            gyro = np.array([msg.data.angular_velocity.x,
                            msg.data.angular_velocity.y,
                            msg.data.angular_velocity.z])
            bias_comp = msg.bias_comp
            imu_status = msg.imu_status
            corrimu_data.append((ts, acc, gyro, bias_comp, imu_status))

        elif topic == '/imu/data':
            ts = msg.header.stamp.to_sec()
            acc = np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z])
            gyro = np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
            imu_data.append((ts, acc, gyro))

    bag.close()

    print(f"\nCORRIMU messages in first 5s: {len(corrimu_data)}")
    print(f"/imu/data messages in first 5s: {len(imu_data)}")

    if corrimu_data:
        print(f"\nCORRIMU bias_comp flag: {corrimu_data[0][3]}")
        print(f"CORRIMU imu_status: {corrimu_data[0][4]}")

    # Calculate mean acceleration (should be ~gravity when stationary)
    corrimu_accs = np.array([msg[1] for msg in corrimu_data])
    imu_accs = np.array([msg[1] for msg in imu_data])
    corrimu_gyros = np.array([msg[2] for msg in corrimu_data])
    imu_gyros = np.array([msg[2] for msg in imu_data])

    print("\n" + "="*60)
    print("STATIONARY ANALYSIS (first 5 seconds)")
    print("="*60)

    print("\nMean Acceleration:")
    corrimu_acc_mean = np.mean(corrimu_accs, axis=0)
    imu_acc_mean = np.mean(imu_accs, axis=0)
    print(f"  CORRIMU:  X={corrimu_acc_mean[0]:+.4f}, Y={corrimu_acc_mean[1]:+.4f}, Z={corrimu_acc_mean[2]:+.4f}")
    print(f"  /imu/data: X={imu_acc_mean[0]:+.4f}, Y={imu_acc_mean[1]:+.4f}, Z={imu_acc_mean[2]:+.4f}")
    print(f"  Diff:      X={corrimu_acc_mean[0]-imu_acc_mean[0]:+.4f}, Y={corrimu_acc_mean[1]-imu_acc_mean[1]:+.4f}, Z={corrimu_acc_mean[2]-imu_acc_mean[2]:+.4f}")

    print("\nMean Gyroscope:")
    corrimu_gyro_mean = np.mean(corrimu_gyros, axis=0)
    imu_gyro_mean = np.mean(imu_gyros, axis=0)
    print(f"  CORRIMU:  X={corrimu_gyro_mean[0]:+.6f}, Y={corrimu_gyro_mean[1]:+.6f}, Z={corrimu_gyro_mean[2]:+.6f}")
    print(f"  /imu/data: X={imu_gyro_mean[0]:+.6f}, Y={imu_gyro_mean[1]:+.6f}, Z={imu_gyro_mean[2]:+.6f}")
    print(f"  Diff:      X={corrimu_gyro_mean[0]-imu_gyro_mean[0]:+.6f}, Y={corrimu_gyro_mean[1]-imu_gyro_mean[1]:+.6f}, Z={corrimu_gyro_mean[2]-imu_gyro_mean[2]:+.6f}")

    # Check bias_comp flag
    if corrimu_data:
        all_bias_comp = [msg[3] for msg in corrimu_data]
        all_imu_status = [msg[4] for msg in corrimu_data]
        print(f"\nCORRIMU bias_comp values: {set(all_bias_comp)}")
        print(f"CORRIMU imu_status values: {set(all_imu_status)}")

    # Check if difference could be just Rz(90) rotation
    print("\n" + "="*60)
    print("COORDINATE FRAME CHECK")
    print("="*60)
    
    # If CORRIMU and /imu/data are in different frames, applying Rz(90) should make them match
    angle = np.radians(90)
    Rz90 = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])
    
    # Transform CORRIMU to /imu/data frame
    corrimu_acc_rotated = Rz90 @ corrimu_acc_mean
    corrimu_gyro_rotated = Rz90 @ corrimu_gyro_mean
    
    print("\nAfter applying Rz(90) to CORRIMU:")
    print(f"  CORRIMU*Rz(90):  X={corrimu_acc_rotated[0]:+.4f}, Y={corrimu_acc_rotated[1]:+.4f}, Z={corrimu_acc_rotated[2]:+.4f}")
    print(f"  /imu/data:       X={imu_acc_mean[0]:+.4f}, Y={imu_acc_mean[1]:+.4f}, Z={imu_acc_mean[2]:+.4f}")
    print(f"  Diff:            X={corrimu_acc_rotated[0]-imu_acc_mean[0]:+.4f}, Y={corrimu_acc_rotated[1]-imu_acc_mean[1]:+.4f}, Z={corrimu_acc_rotated[2]-imu_acc_mean[2]:+.4f}")


if __name__ == "__main__":
    main()
