#!/usr/bin/env python3
"""
Diagnostic script to analyze IMU data from bag file and trace the preintegration.
This helps identify the root cause of velocity drift.
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Parameters matching params.yaml
IMU_GRAVITY = 9.80511
EXT_ROT = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Rz(180)
EXT_RPY = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])  # Rz(180)

def quaternion_to_euler(q):
    """Convert quaternion [x,y,z,w] to euler angles [roll, pitch, yaw] in radians"""
    r = R.from_quat(q)
    return r.as_euler('xyz', degrees=False)

def euler_to_quaternion(euler):
    """Convert euler angles [roll, pitch, yaw] to quaternion [x,y,z,w]"""
    r = R.from_euler('xyz', euler)
    return r.as_quat()

def imu_converter(acc, gyr, orientation):
    """Convert IMU data from IMU frame to LiDAR frame (matches utility.h imuConverter)"""
    # Rotate acceleration
    acc_lidar = EXT_ROT @ acc

    # Rotate gyroscope
    gyr_lidar = EXT_ROT @ gyr

    # Rotate orientation
    # extQRPY = inverse(Quaternion(extRPY))
    # q_final = q_from * extQRPY
    r_from = R.from_quat(orientation)  # [x,y,z,w] format
    r_ext = R.from_matrix(EXT_RPY)
    r_final = r_from * r_ext.inv()  # This is q_from * extQRPY
    orientation_lidar = r_final.as_quat()  # [x,y,z,w] format

    return acc_lidar, gyr_lidar, orientation_lidar

def simulate_preintegration(acc_samples, gyr_samples, dt_samples, initial_orientation):
    """
    Simulate simplified preintegration to understand velocity drift.

    This uses the formula from GTSAM's MakeSharedU convention:
    a_nav = R_nav_body * (a_body - bias) + g_nav
    where g_nav = [0, 0, -g] (gravity pointing down, Z-up)
    """
    g_nav = np.array([0, 0, -IMU_GRAVITY])
    bias_a = np.array([0, 0, 0])  # Assume zero bias

    velocity = np.array([0.0, 0.0, 0.0])
    position = np.array([0.0, 0.0, 0.0])

    # Use initial orientation
    r_nav_body = R.from_quat(initial_orientation)

    velocities = [velocity.copy()]
    positions = [position.copy()]
    raw_accels = []
    corrected_accels = []

    for i, (acc, gyr, dt) in enumerate(zip(acc_samples, gyr_samples, dt_samples)):
        # Compute acceleration in nav frame (gravity compensated)
        a_nav = r_nav_body.as_matrix() @ (acc - bias_a) + g_nav

        # Store for analysis
        raw_accels.append(acc.copy())
        corrected_accels.append(a_nav.copy())

        # Integrate velocity and position
        velocity = velocity + a_nav * dt
        position = position + velocity * dt + 0.5 * a_nav * dt * dt

        velocities.append(velocity.copy())
        positions.append(position.copy())

        # Update orientation using gyroscope
        # delta_R = exp(gyr * dt)
        if np.linalg.norm(gyr) > 1e-10:
            delta_r = R.from_rotvec(gyr * dt)
            r_nav_body = r_nav_body * delta_r

    return np.array(velocities), np.array(positions), np.array(raw_accels), np.array(corrected_accels)


def main():
    bag_path = '/root/autodl-tmp/info_fixed.bag'
    imu_topic = '/imu/data'

    print(f"Reading IMU data from {bag_path}...")

    imu_data = []
    bag = rosbag.Bag(bag_path, 'r')

    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]),
            'gyr': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        })

    bag.close()
    print(f"Read {len(imu_data)} IMU messages")

    if len(imu_data) == 0:
        print("No IMU data found!")
        return

    # Analyze first 10 seconds
    start_time = imu_data[0]['time']
    analysis_duration = 10.0

    print(f"\n=== Analyzing first {analysis_duration} seconds ===")

    # Filter to first 10 seconds
    imu_first_10s = [d for d in imu_data if d['time'] - start_time <= analysis_duration]
    print(f"IMU messages in first {analysis_duration}s: {len(imu_first_10s)}")

    # Check raw IMU values
    print("\n--- Raw IMU Statistics ---")
    accs_raw = np.array([d['acc'] for d in imu_first_10s])
    gyrs_raw = np.array([d['gyr'] for d in imu_first_10s])

    print(f"Raw Acceleration (m/s^2):")
    print(f"  Mean: {np.mean(accs_raw, axis=0)}")
    print(f"  Std:  {np.std(accs_raw, axis=0)}")
    print(f"  Expected stationary: [0, 0, {IMU_GRAVITY}] (if +X forward, Z up)")

    print(f"\nRaw Gyroscope (rad/s):")
    print(f"  Mean: {np.mean(gyrs_raw, axis=0)}")
    print(f"  Std:  {np.std(gyrs_raw, axis=0)}")
    print(f"  Expected stationary: [0, 0, 0]")

    # Check orientation
    print("\n--- Orientation Analysis ---")
    orientations = np.array([d['orientation'] for d in imu_first_10s])
    first_euler = quaternion_to_euler(orientations[0]) * 180 / np.pi
    last_euler = quaternion_to_euler(orientations[-1]) * 180 / np.pi
    print(f"First orientation (r,p,y deg): {first_euler}")
    print(f"Last orientation (r,p,y deg): {last_euler}")
    print(f"Change: {last_euler - first_euler}")

    # Convert to LiDAR frame
    print("\n--- Converted IMU (LiDAR frame) ---")
    converted_data = []
    for d in imu_first_10s:
        acc_l, gyr_l, ori_l = imu_converter(d['acc'], d['gyr'], d['orientation'])
        converted_data.append({
            'time': d['time'],
            'acc': acc_l,
            'gyr': gyr_l,
            'orientation': ori_l
        })

    accs_lidar = np.array([d['acc'] for d in converted_data])
    gyrs_lidar = np.array([d['gyr'] for d in converted_data])

    print(f"Converted Acceleration (m/s^2):")
    print(f"  Mean: {np.mean(accs_lidar, axis=0)}")
    print(f"  Std:  {np.std(accs_lidar, axis=0)}")
    print(f"  Expected stationary: [0, 0, {IMU_GRAVITY}]")

    first_euler_l = quaternion_to_euler(converted_data[0]['orientation']) * 180 / np.pi
    print(f"\nFirst converted orientation (r,p,y deg): {first_euler_l}")

    # Simulate preintegration
    print("\n--- Simulating Preintegration ---")

    # Prepare data
    acc_samples = [d['acc'] for d in converted_data]
    gyr_samples = [d['gyr'] for d in converted_data]
    times = [d['time'] for d in converted_data]
    dt_samples = [times[i+1] - times[i] if i < len(times)-1 else 0.005 for i in range(len(times))]

    # Use first orientation as initial state
    initial_orientation = converted_data[0]['orientation']

    velocities, positions, raw_accels, corrected_accels = simulate_preintegration(
        acc_samples, gyr_samples, dt_samples, initial_orientation
    )

    print(f"\nFinal velocity after {analysis_duration}s: {velocities[-1]}")
    print(f"Final position after {analysis_duration}s: {positions[-1]}")
    print(f"Velocity magnitude: {np.linalg.norm(velocities[-1]):.2f} m/s")

    # Analyze corrected acceleration (should be near zero for stationary)
    print("\n--- Gravity-Corrected Acceleration (should be ~0 when stationary) ---")
    corrected_accels = np.array(corrected_accels)
    print(f"Mean corrected accel: {np.mean(corrected_accels, axis=0)}")
    print(f"This represents average uncompensated acceleration")

    mean_corrected = np.mean(corrected_accels, axis=0)
    if np.linalg.norm(mean_corrected) > 0.1:
        print(f"\nWARNING: Large uncompensated acceleration detected!")
        print(f"Possible causes:")
        print(f"  1. Accelerometer bias not compensated")
        print(f"  2. Orientation error causing gravity misalignment")
        print(f"  3. Vehicle was actually moving")

        # Estimate orientation error needed to cause this
        # If mean_corrected[0] or [1] is non-zero, it suggests orientation error
        horizontal_accel = np.sqrt(mean_corrected[0]**2 + mean_corrected[1]**2)
        if horizontal_accel > 0.01:
            estimated_tilt_error = np.arcsin(horizontal_accel / IMU_GRAVITY) * 180 / np.pi
            print(f"\nEstimated orientation error: ~{estimated_tilt_error:.1f} degrees")

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    times_plot = np.array(times) - times[0]

    # Raw acceleration
    axes[0, 0].plot(times_plot, accs_raw[:, 0], label='X', alpha=0.7)
    axes[0, 0].plot(times_plot, accs_raw[:, 1], label='Y', alpha=0.7)
    axes[0, 0].plot(times_plot, accs_raw[:, 2], label='Z', alpha=0.7)
    axes[0, 0].axhline(y=IMU_GRAVITY, color='k', linestyle='--', label=f'g={IMU_GRAVITY}')
    axes[0, 0].set_ylabel('m/s^2')
    axes[0, 0].set_title('Raw IMU Acceleration')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Corrected acceleration
    axes[0, 1].plot(times_plot, corrected_accels[:, 0], label='X', alpha=0.7)
    axes[0, 1].plot(times_plot, corrected_accels[:, 1], label='Y', alpha=0.7)
    axes[0, 1].plot(times_plot, corrected_accels[:, 2], label='Z', alpha=0.7)
    axes[0, 1].axhline(y=0, color='k', linestyle='--')
    axes[0, 1].set_ylabel('m/s^2')
    axes[0, 1].set_title('Gravity-Corrected Acceleration (should be ~0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Velocity
    axes[1, 0].plot(times_plot, velocities[:-1, 0], label='X', alpha=0.7)
    axes[1, 0].plot(times_plot, velocities[:-1, 1], label='Y', alpha=0.7)
    axes[1, 0].plot(times_plot, velocities[:-1, 2], label='Z', alpha=0.7)
    axes[1, 0].set_ylabel('m/s')
    axes[1, 0].set_title('Integrated Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Position
    axes[1, 1].plot(times_plot, positions[:-1, 0], label='X', alpha=0.7)
    axes[1, 1].plot(times_plot, positions[:-1, 1], label='Y', alpha=0.7)
    axes[1, 1].plot(times_plot, positions[:-1, 2], label='Z', alpha=0.7)
    axes[1, 1].set_ylabel('m')
    axes[1, 1].set_title('Integrated Position')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Orientation (euler angles)
    eulers_raw = np.array([quaternion_to_euler(d['orientation']) for d in imu_first_10s]) * 180 / np.pi
    axes[2, 0].plot(times_plot, eulers_raw[:, 0], label='Roll', alpha=0.7)
    axes[2, 0].plot(times_plot, eulers_raw[:, 1], label='Pitch', alpha=0.7)
    axes[2, 0].plot(times_plot, eulers_raw[:, 2], label='Yaw', alpha=0.7)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('degrees')
    axes[2, 0].set_title('Raw IMU Orientation')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Converted orientation
    eulers_conv = np.array([quaternion_to_euler(d['orientation']) for d in converted_data]) * 180 / np.pi
    axes[2, 1].plot(times_plot, eulers_conv[:, 0], label='Roll', alpha=0.7)
    axes[2, 1].plot(times_plot, eulers_conv[:, 1], label='Pitch', alpha=0.7)
    axes[2, 1].plot(times_plot, eulers_conv[:, 2], label='Yaw', alpha=0.7)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('degrees')
    axes[2, 1].set_title('Converted Orientation (LiDAR frame)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/catkin_ws/src/LIO-SAM/output/preintegration_analysis.png', dpi=150)
    print(f"\nPlot saved to /root/autodl-tmp/catkin_ws/src/LIO-SAM/output/preintegration_analysis.png")
    plt.close()


if __name__ == '__main__':
    main()
