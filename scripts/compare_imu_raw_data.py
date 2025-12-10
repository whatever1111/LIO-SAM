#!/usr/bin/env python3
"""
Compare raw IMU data between CORRIMU and /imu/data to find differences
that could cause divergence in FPA mode.
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R

BAG_FILE = "/root/autodl-tmp/info2.bag"

def main():
    bag = rosbag.Bag(BAG_FILE)

    # Collect data
    corrimu_data = []  # (time, acc, gyro)
    imu_data = []      # (time, acc, gyro)

    print("Reading bag file...")

    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/corrimu', '/imu/data']):
        if topic == '/fixposition/fpa/corrimu':
            ts = msg.data.header.stamp.to_sec()
            acc = np.array([msg.data.linear_acceleration.x,
                           msg.data.linear_acceleration.y,
                           msg.data.linear_acceleration.z])
            gyro = np.array([msg.data.angular_velocity.x,
                            msg.data.angular_velocity.y,
                            msg.data.angular_velocity.z])
            corrimu_data.append((ts, acc, gyro))

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

    print(f"CORRIMU messages: {len(corrimu_data)}")
    print(f"/imu/data messages: {len(imu_data)}")

    if not corrimu_data or not imu_data:
        print("No data found!")
        return

    # Time alignment
    corrimu_start = corrimu_data[0][0]
    imu_start = imu_data[0][0]

    print(f"\nCORRIMU start time: {corrimu_start:.6f}")
    print(f"/imu/data start time: {imu_start:.6f}")
    print(f"Time difference: {(corrimu_start - imu_start)*1000:.3f} ms")

    # Compare first 10 seconds of data
    print("\n" + "="*80)
    print("Comparing first 10 seconds of data")
    print("="*80)

    # extrinsicRot = Rz(90)
    angle = np.radians(90)
    extRot = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    # Find closest /imu/data message for each CORRIMU message
    imu_idx = 0

    print("\nComparing raw data (before transformation):")
    print("-" * 80)

    # Sample comparison at different times
    sample_times = [0, 5, 10, 20, 30, 40, 45]  # seconds from start

    for t_offset in sample_times:
        target_time = corrimu_start + t_offset

        # Find closest CORRIMU message
        corrimu_msg = None
        for ts, acc, gyro in corrimu_data:
            if ts >= target_time:
                corrimu_msg = (ts, acc, gyro)
                break

        if corrimu_msg is None:
            print(f"\nt={t_offset}s: No CORRIMU data")
            continue

        # Find closest /imu/data message
        imu_msg = None
        min_dt = float('inf')
        for ts, acc, gyro in imu_data:
            dt = abs(ts - corrimu_msg[0])
            if dt < min_dt:
                min_dt = dt
                imu_msg = (ts, acc, gyro)
            if ts > corrimu_msg[0] + 0.1:
                break

        if imu_msg is None:
            print(f"\nt={t_offset}s: No /imu/data")
            continue

        print(f"\n=== t = {t_offset}s (actual t={corrimu_msg[0]:.3f}) ===")
        print(f"Time diff between messages: {(corrimu_msg[0] - imu_msg[0])*1000:.2f} ms")

        # Raw acceleration
        print(f"\nRaw Acceleration (m/s^2):")
        print(f"  CORRIMU:  X={corrimu_msg[1][0]:+8.4f}, Y={corrimu_msg[1][1]:+8.4f}, Z={corrimu_msg[1][2]:+8.4f}")
        print(f"  /imu/data: X={imu_msg[1][0]:+8.4f}, Y={imu_msg[1][1]:+8.4f}, Z={imu_msg[1][2]:+8.4f}")
        print(f"  Diff:      X={corrimu_msg[1][0]-imu_msg[1][0]:+8.4f}, Y={corrimu_msg[1][1]-imu_msg[1][1]:+8.4f}, Z={corrimu_msg[1][2]-imu_msg[1][2]:+8.4f}")

        # Acceleration magnitude (should be ~9.8 when stationary)
        corrimu_acc_mag = np.linalg.norm(corrimu_msg[1])
        imu_acc_mag = np.linalg.norm(imu_msg[1])
        print(f"\n  CORRIMU acc magnitude:  {corrimu_acc_mag:.4f}")
        print(f"  /imu/data acc magnitude: {imu_acc_mag:.4f}")

        # Transformed acceleration
        corrimu_acc_trans = extRot @ corrimu_msg[1]
        imu_acc_trans = extRot @ imu_msg[1]
        print(f"\nAfter extRot transformation:")
        print(f"  CORRIMU:  X={corrimu_acc_trans[0]:+8.4f}, Y={corrimu_acc_trans[1]:+8.4f}, Z={corrimu_acc_trans[2]:+8.4f}")
        print(f"  /imu/data: X={imu_acc_trans[0]:+8.4f}, Y={imu_acc_trans[1]:+8.4f}, Z={imu_acc_trans[2]:+8.4f}")

        # Raw gyroscope
        print(f"\nRaw Angular Velocity (rad/s):")
        print(f"  CORRIMU:  X={corrimu_msg[2][0]:+8.5f}, Y={corrimu_msg[2][1]:+8.5f}, Z={corrimu_msg[2][2]:+8.5f}")
        print(f"  /imu/data: X={imu_msg[2][0]:+8.5f}, Y={imu_msg[2][1]:+8.5f}, Z={imu_msg[2][2]:+8.5f}")
        print(f"  Diff:      X={corrimu_msg[2][0]-imu_msg[2][0]:+8.5f}, Y={corrimu_msg[2][1]-imu_msg[2][1]:+8.5f}, Z={corrimu_msg[2][2]-imu_msg[2][2]:+8.5f}")

        # Transformed gyroscope
        corrimu_gyro_trans = extRot @ corrimu_msg[2]
        imu_gyro_trans = extRot @ imu_msg[2]
        print(f"\nAfter extRot transformation:")
        print(f"  CORRIMU:  X={corrimu_gyro_trans[0]:+8.5f}, Y={corrimu_gyro_trans[1]:+8.5f}, Z={corrimu_gyro_trans[2]:+8.5f}")
        print(f"  /imu/data: X={imu_gyro_trans[0]:+8.5f}, Y={imu_gyro_trans[1]:+8.5f}, Z={imu_gyro_trans[2]:+8.5f}")

    # Statistics over time
    print("\n" + "="*80)
    print("Statistical comparison (first 50 seconds)")
    print("="*80)

    end_time = corrimu_start + 50
    corrimu_50s = [(ts, acc, gyro) for ts, acc, gyro in corrimu_data if ts < end_time]
    imu_50s = [(ts, acc, gyro) for ts, acc, gyro in imu_data if ts < end_time]

    corrimu_accs = np.array([msg[1] for msg in corrimu_50s])
    imu_accs = np.array([msg[1] for msg in imu_50s])
    corrimu_gyros = np.array([msg[2] for msg in corrimu_50s])
    imu_gyros = np.array([msg[2] for msg in imu_50s])

    print("\nAcceleration statistics (raw):")
    print(f"  CORRIMU mean:  X={np.mean(corrimu_accs[:,0]):+8.4f}, Y={np.mean(corrimu_accs[:,1]):+8.4f}, Z={np.mean(corrimu_accs[:,2]):+8.4f}")
    print(f"  /imu/data mean: X={np.mean(imu_accs[:,0]):+8.4f}, Y={np.mean(imu_accs[:,1]):+8.4f}, Z={np.mean(imu_accs[:,2]):+8.4f}")
    print(f"  CORRIMU std:   X={np.std(corrimu_accs[:,0]):+8.4f}, Y={np.std(corrimu_accs[:,1]):+8.4f}, Z={np.std(corrimu_accs[:,2]):+8.4f}")
    print(f"  /imu/data std:  X={np.std(imu_accs[:,0]):+8.4f}, Y={np.std(imu_accs[:,1]):+8.4f}, Z={np.std(imu_accs[:,2]):+8.4f}")

    print("\nGyroscope statistics (raw):")
    print(f"  CORRIMU mean:  X={np.mean(corrimu_gyros[:,0]):+8.5f}, Y={np.mean(corrimu_gyros[:,1]):+8.5f}, Z={np.mean(corrimu_gyros[:,2]):+8.5f}")
    print(f"  /imu/data mean: X={np.mean(imu_gyros[:,0]):+8.5f}, Y={np.mean(imu_gyros[:,1]):+8.5f}, Z={np.mean(imu_gyros[:,2]):+8.5f}")
    print(f"  CORRIMU std:   X={np.std(corrimu_gyros[:,0]):+8.5f}, Y={np.std(corrimu_gyros[:,1]):+8.5f}, Z={np.std(corrimu_gyros[:,2]):+8.5f}")
    print(f"  /imu/data std:  X={np.std(imu_gyros[:,0]):+8.5f}, Y={np.std(imu_gyros[:,1]):+8.5f}, Z={np.std(imu_gyros[:,2]):+8.5f}")

    # Check if data is actually different
    print("\n" + "="*80)
    print("KEY FINDING: Are CORRIMU and /imu/data actually different?")
    print("="*80)

    # Find matching timestamps
    matched_pairs = []
    imu_dict = {ts: (acc, gyro) for ts, acc, gyro in imu_data}

    for ts, acc, gyro in corrimu_data[:1000]:  # First 1000 messages
        # Find closest /imu/data
        closest_ts = min(imu_dict.keys(), key=lambda x: abs(x - ts))
        if abs(closest_ts - ts) < 0.01:  # Within 10ms
            matched_pairs.append((ts, acc, gyro, imu_dict[closest_ts][0], imu_dict[closest_ts][1]))

    if matched_pairs:
        acc_diffs = []
        gyro_diffs = []
        for ts, c_acc, c_gyro, i_acc, i_gyro in matched_pairs:
            acc_diffs.append(c_acc - i_acc)
            gyro_diffs.append(c_gyro - i_gyro)

        acc_diffs = np.array(acc_diffs)
        gyro_diffs = np.array(gyro_diffs)

        print(f"\nMatched {len(matched_pairs)} message pairs")
        print(f"\nAcceleration difference (CORRIMU - /imu/data):")
        print(f"  Mean: X={np.mean(acc_diffs[:,0]):+.6f}, Y={np.mean(acc_diffs[:,1]):+.6f}, Z={np.mean(acc_diffs[:,2]):+.6f}")
        print(f"  Max:  X={np.max(np.abs(acc_diffs[:,0])):.6f}, Y={np.max(np.abs(acc_diffs[:,1])):.6f}, Z={np.max(np.abs(acc_diffs[:,2])):.6f}")

        print(f"\nGyroscope difference (CORRIMU - /imu/data):")
        print(f"  Mean: X={np.mean(gyro_diffs[:,0]):+.7f}, Y={np.mean(gyro_diffs[:,1]):+.7f}, Z={np.mean(gyro_diffs[:,2]):+.7f}")
        print(f"  Max:  X={np.max(np.abs(gyro_diffs[:,0])):.7f}, Y={np.max(np.abs(gyro_diffs[:,1])):.7f}, Z={np.max(np.abs(gyro_diffs[:,2])):.7f}")

        if np.max(np.abs(acc_diffs)) < 0.001 and np.max(np.abs(gyro_diffs)) < 0.00001:
            print("\n>>> CORRIMU and /imu/data have IDENTICAL acc/gyro data! <<<")
        else:
            print("\n>>> CORRIMU and /imu/data have DIFFERENT acc/gyro data <<<")


if __name__ == "__main__":
    main()
