#!/usr/bin/env python3
"""
Compare /imu/data vs /fixposition/fpa/corrimu
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R

IMU_GRAVITY = 9.80511

def analyze_standard_imu(bag_path, topic_name, duration=10.0):
    """Analyze standard sensor_msgs/Imu topic"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {topic_name}")
    print(f"{'='*60}")

    imu_data = []
    bag = rosbag.Bag(bag_path, 'r')

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]),
            'gyr': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        })

    bag.close()
    return analyze_imu_data(imu_data, topic_name, duration)


def analyze_fpa_imu(bag_path, topic_name, duration=10.0):
    """Analyze FpaImu topic (fixposition custom message)"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {topic_name}")
    print(f"{'='*60}")

    imu_data = []
    bias_comp_values = []
    imu_status_values = []
    bag = rosbag.Bag(bag_path, 'r')

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        # FpaImu has: bias_comp, imu_status, data (sensor_msgs/Imu)
        data = msg.data
        imu_data.append({
            'time': data.header.stamp.to_sec(),
            'acc': np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z]),
            'gyr': np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z]),
            'orientation': np.array([data.orientation.x, data.orientation.y, data.orientation.z, data.orientation.w])
        })
        bias_comp_values.append(msg.bias_comp)
        imu_status_values.append(msg.imu_status)

    bag.close()

    if len(imu_data) > 0:
        print(f"bias_comp: {bias_comp_values[0]} (True=bias compensated)")
        print(f"imu_status: {imu_status_values[0]} (0=not converged, 1=converged)")
        # Check if status changes
        if len(set(imu_status_values)) > 1:
            print(f"  Status changes during recording!")

    return analyze_imu_data(imu_data, topic_name, duration)


def analyze_imu_data(imu_data, topic_name, duration):
    """Common analysis for IMU data"""
    if len(imu_data) == 0:
        print(f"No data found on topic {topic_name}")
        return None

    print(f"Total messages: {len(imu_data)}")

    # Filter to first N seconds
    start_time = imu_data[0]['time']
    imu_filtered = [d for d in imu_data if d['time'] - start_time <= duration]
    print(f"Messages in first {duration}s: {len(imu_filtered)}")

    if len(imu_filtered) < 10:
        print("Not enough data")
        return None

    # Statistics
    accs = np.array([d['acc'] for d in imu_filtered])
    gyrs = np.array([d['gyr'] for d in imu_filtered])

    print(f"\nAcceleration (m/s^2):")
    print(f"  Mean: [{accs[:,0].mean():.6f}, {accs[:,1].mean():.6f}, {accs[:,2].mean():.6f}]")
    print(f"  Std:  [{accs[:,0].std():.6f}, {accs[:,1].std():.6f}, {accs[:,2].std():.6f}]")
    print(f"  Expected (stationary, Z-up): [0, 0, ~{IMU_GRAVITY}]")

    # Check if gravity is in Z
    acc_mean = accs.mean(axis=0)
    acc_norm = np.linalg.norm(acc_mean)
    print(f"  Magnitude: {acc_norm:.4f} (expected ~{IMU_GRAVITY})")

    # Estimate tilt from acceleration
    if acc_norm > 1:
        tilt_from_z = np.arccos(abs(acc_mean[2]) / acc_norm) * 180 / np.pi
        print(f"  Tilt from vertical: {tilt_from_z:.2f} degrees")

    print(f"\nGyroscope (rad/s):")
    print(f"  Mean: [{gyrs[:,0].mean():.6f}, {gyrs[:,1].mean():.6f}, {gyrs[:,2].mean():.6f}]")
    print(f"  Std:  [{gyrs[:,0].std():.6f}, {gyrs[:,1].std():.6f}, {gyrs[:,2].std():.6f}]")
    print(f"  Expected (stationary): [0, 0, 0]")

    # Orientation
    ori_first = imu_filtered[0]['orientation']
    ori_last = imu_filtered[-1]['orientation']

    # Check if orientation is valid (non-zero)
    ori_norm = np.linalg.norm(ori_first)
    print(f"\nOrientation:")
    print(f"  First quaternion [x,y,z,w]: {ori_first}")
    print(f"  Quaternion norm: {ori_norm:.4f} (should be ~1.0)")

    if ori_norm > 0.1:
        r = R.from_quat(ori_first)
        euler = r.as_euler('xyz', degrees=True)
        print(f"  First Euler (r,p,y) deg: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")

        r_last = R.from_quat(ori_last)
        euler_last = r_last.as_euler('xyz', degrees=True)
        print(f"  Last Euler (r,p,y) deg: [{euler_last[0]:.2f}, {euler_last[1]:.2f}, {euler_last[2]:.2f}]")
        print(f"  Change: [{euler_last[0]-euler[0]:.4f}, {euler_last[1]-euler[1]:.4f}, {euler_last[2]-euler[2]:.4f}]")
    else:
        print(f"  WARNING: Invalid orientation (norm too small)")

    # Compute bias (deviation from expected)
    print(f"\nBias Analysis:")
    acc_bias_xy = np.sqrt(acc_mean[0]**2 + acc_mean[1]**2)
    print(f"  Horizontal acc (X-Y plane): {acc_bias_xy:.4f} m/s^2")
    print(f"  Z acc deviation from g: {acc_mean[2] - IMU_GRAVITY:.4f} m/s^2")

    gyr_bias = np.linalg.norm(gyrs.mean(axis=0))
    print(f"  Gyro bias magnitude: {gyr_bias:.6f} rad/s ({gyr_bias*180/np.pi:.4f} deg/s)")

    return {
        'acc_mean': acc_mean,
        'gyr_mean': gyrs.mean(axis=0),
        'acc_std': accs.std(axis=0),
        'gyr_std': gyrs.std(axis=0),
        'ori_first': ori_first
    }


def main():
    bag_path = '/root/autodl-tmp/info_fixed.bag'

    # Analyze both topics
    result_imu = analyze_standard_imu(bag_path, '/imu/data')
    result_corrimu = analyze_fpa_imu(bag_path, '/fixposition/fpa/corrimu')

    # Compare if both available
    if result_imu and result_corrimu:
        print(f"\n{'='*60}")
        print("COMPARISON: /imu/data vs /fixposition/fpa/corrimu")
        print(f"{'='*60}")

        print(f"\nAcceleration bias (horizontal):")
        bias_imu = np.sqrt(result_imu['acc_mean'][0]**2 + result_imu['acc_mean'][1]**2)
        bias_corr = np.sqrt(result_corrimu['acc_mean'][0]**2 + result_corrimu['acc_mean'][1]**2)
        print(f"  /imu/data:                 {bias_imu:.4f} m/s^2")
        print(f"  /fixposition/fpa/corrimu:  {bias_corr:.4f} m/s^2")
        if bias_imu > 0:
            improvement = (1 - bias_corr/bias_imu)*100
            print(f"  Improvement: {improvement:.1f}%")

        print(f"\nGyroscope bias:")
        gyr_bias_imu = np.linalg.norm(result_imu['gyr_mean'])
        gyr_bias_corr = np.linalg.norm(result_corrimu['gyr_mean'])
        print(f"  /imu/data:                 {gyr_bias_imu:.6f} rad/s")
        print(f"  /fixposition/fpa/corrimu:  {gyr_bias_corr:.6f} rad/s")

        print(f"\nAcceleration noise (std):")
        print(f"  /imu/data:                 {result_imu['acc_std']}")
        print(f"  /fixposition/fpa/corrimu:  {result_corrimu['acc_std']}")

        print(f"\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        if bias_corr < bias_imu * 0.5:
            print("  -> /fixposition/fpa/corrimu has SIGNIFICANTLY LOWER bias")
            print("  -> STRONGLY RECOMMEND using corrimu for LIO-SAM")
        elif bias_corr < bias_imu:
            print("  -> /fixposition/fpa/corrimu has slightly lower bias")
            print("  -> Consider using corrimu for better performance")
        else:
            print("  -> Both topics have similar bias levels")


if __name__ == '__main__':
    main()
