#!/usr/bin/env python3
"""
Test fused IMU data quality by simulating preintegration
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R

IMU_GRAVITY = 9.80511
EXT_ROT = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Rz(180)

def imu_converter(acc, gyr, orientation):
    """Convert IMU data from IMU frame to LiDAR frame"""
    acc_lidar = EXT_ROT @ acc
    gyr_lidar = EXT_ROT @ gyr

    r_from = R.from_quat(orientation)
    r_ext = R.from_matrix(EXT_ROT)
    r_final = r_from * r_ext.inv()
    orientation_lidar = r_final.as_quat()

    return acc_lidar, gyr_lidar, orientation_lidar


def simulate_preintegration(acc_samples, gyr_samples, dt_samples, initial_orientation):
    """Simulate preintegration"""
    g_nav = np.array([0, 0, -IMU_GRAVITY])
    bias_a = np.array([0, 0, 0])

    velocity = np.array([0.0, 0.0, 0.0])
    r_nav_body = R.from_quat(initial_orientation)

    for acc, gyr, dt in zip(acc_samples, gyr_samples, dt_samples):
        a_nav = r_nav_body.as_matrix() @ (acc - bias_a) + g_nav
        velocity = velocity + a_nav * dt
        if np.linalg.norm(gyr) > 1e-10:
            delta_r = R.from_rotvec(gyr * dt)
            r_nav_body = r_nav_body * delta_r

    return velocity


def analyze_topic(bag_path, imu_topic, corrimu_topic, duration=10.0):
    """Create fused data and analyze"""

    print(f"\n{'='*60}")
    print("Simulating Fused IMU: orientation from /imu/data")
    print("                     + gyro/accel from /fixposition/fpa/corrimu")
    print(f"{'='*60}")

    # Read both topics
    imu_data = {}
    corrimu_data = []

    bag = rosbag.Bag(bag_path, 'r')

    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        time_key = round(msg.header.stamp.to_sec(), 4)
        imu_data[time_key] = {
            'orientation': np.array([msg.orientation.x, msg.orientation.y,
                                    msg.orientation.z, msg.orientation.w])
        }

    for topic, msg, t in bag.read_messages(topics=[corrimu_topic]):
        data = msg.data
        corrimu_data.append({
            'time': data.header.stamp.to_sec(),
            'acc': np.array([data.linear_acceleration.x, data.linear_acceleration.y,
                           data.linear_acceleration.z]),
            'gyr': np.array([data.angular_velocity.x, data.angular_velocity.y,
                           data.angular_velocity.z])
        })

    bag.close()

    print(f"Loaded {len(imu_data)} /imu/data messages")
    print(f"Loaded {len(corrimu_data)} /fixposition/fpa/corrimu messages")

    # Filter to first N seconds
    start_time = corrimu_data[0]['time']
    corrimu_filtered = [d for d in corrimu_data if d['time'] - start_time <= duration]

    # Create fused data
    fused_data = []
    orientation_found = 0
    for d in corrimu_filtered:
        time_key = round(d['time'], 4)
        # Find closest orientation
        best_key = None
        best_diff = 0.1
        for k in imu_data.keys():
            diff = abs(k - time_key)
            if diff < best_diff:
                best_diff = diff
                best_key = k

        if best_key is not None:
            ori = imu_data[best_key]['orientation']
            if np.linalg.norm(ori) > 0.9:
                fused_data.append({
                    'time': d['time'],
                    'acc': d['acc'],
                    'gyr': d['gyr'],
                    'orientation': ori
                })
                orientation_found += 1

    print(f"Fused messages with valid orientation: {len(fused_data)}")

    if len(fused_data) < 100:
        print("Not enough fused data")
        return

    # Statistics
    accs = np.array([d['acc'] for d in fused_data])
    gyrs = np.array([d['gyr'] for d in fused_data])

    print(f"\nFused IMU Statistics:")
    print(f"  Acc Mean: {accs.mean(axis=0)}")
    print(f"  Gyr Mean: {gyrs.mean(axis=0)}")
    print(f"  Gyr Bias: {np.linalg.norm(gyrs.mean(axis=0)):.6f} rad/s")

    # Convert to LiDAR frame and simulate preintegration
    converted_data = []
    for d in fused_data:
        acc_l, gyr_l, ori_l = imu_converter(d['acc'], d['gyr'], d['orientation'])
        converted_data.append({
            'time': d['time'],
            'acc': acc_l,
            'gyr': gyr_l,
            'orientation': ori_l
        })

    # Preintegration
    acc_samples = [d['acc'] for d in converted_data]
    gyr_samples = [d['gyr'] for d in converted_data]
    times = [d['time'] for d in converted_data]
    dt_samples = [times[i+1] - times[i] if i < len(times)-1 else 0.005 for i in range(len(times))]

    initial_orientation = converted_data[0]['orientation']
    final_velocity = simulate_preintegration(acc_samples, gyr_samples, dt_samples, initial_orientation)

    print(f"\n=== Preintegration Simulation (first {duration}s) ===")
    print(f"Final velocity: {final_velocity}")
    print(f"Velocity magnitude: {np.linalg.norm(final_velocity):.4f} m/s")

    # Compare with original /imu/data
    print(f"\n=== Comparison ===")
    print("Expected velocity for stationary: ~0 m/s")

    return np.linalg.norm(final_velocity)


def analyze_original(bag_path, duration=10.0):
    """Analyze original /imu/data"""
    print(f"\n{'='*60}")
    print("Original /imu/data (for comparison)")
    print(f"{'='*60}")

    imu_data = []
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, t in bag.read_messages(topics=['/imu/data']):
        imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]),
            'gyr': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        })
    bag.close()

    start_time = imu_data[0]['time']
    imu_filtered = [d for d in imu_data if d['time'] - start_time <= duration]

    # Convert to LiDAR frame
    converted_data = []
    for d in imu_filtered:
        acc_l, gyr_l, ori_l = imu_converter(d['acc'], d['gyr'], d['orientation'])
        converted_data.append({
            'time': d['time'],
            'acc': acc_l,
            'gyr': gyr_l,
            'orientation': ori_l
        })

    # Preintegration
    acc_samples = [d['acc'] for d in converted_data]
    gyr_samples = [d['gyr'] for d in converted_data]
    times = [d['time'] for d in converted_data]
    dt_samples = [times[i+1] - times[i] if i < len(times)-1 else 0.005 for i in range(len(times))]

    initial_orientation = converted_data[0]['orientation']
    final_velocity = simulate_preintegration(acc_samples, gyr_samples, dt_samples, initial_orientation)

    print(f"Final velocity: {final_velocity}")
    print(f"Velocity magnitude: {np.linalg.norm(final_velocity):.4f} m/s")

    return np.linalg.norm(final_velocity)


def main():
    bag_path = '/root/autodl-tmp/info_fixed.bag'
    duration = 10.0

    vel_original = analyze_original(bag_path, duration)
    vel_fused = analyze_topic(bag_path, '/imu/data', '/fixposition/fpa/corrimu', duration)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Velocity drift after {duration}s (stationary):")
    print(f"  Original /imu/data:  {vel_original:.4f} m/s")
    print(f"  Fused IMU:           {vel_fused:.4f} m/s")

    if vel_fused < vel_original:
        improvement = (1 - vel_fused/vel_original) * 100
        print(f"\n  Improvement: {improvement:.1f}% less drift with fused IMU")
    else:
        print(f"\n  No improvement with fused IMU")


if __name__ == '__main__':
    main()
