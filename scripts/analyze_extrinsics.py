#!/usr/bin/env python3
"""
Extrinsics Analysis Script
==========================
Analyzes LiDAR and IMU data to determine the correct extrinsic rotation matrix.

This script examines:
1. LiDAR coordinate frame (which axis points forward)
2. IMU gravity direction (which axis is up)
3. IMU orientation during motion

Usage:
    python3 analyze_extrinsics.py /path/to/bag_file.bag
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import sys

def analyze_lidar_frame(bag_path, topic='/lidar_points', num_frames=5):
    """Analyze LiDAR coordinate frame orientation."""
    print("=" * 60)
    print("LiDAR Coordinate Frame Analysis")
    print("=" * 60)

    bag = rosbag.Bag(bag_path)

    all_points = []
    frame_count = 0

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if frame_count >= num_frames:
            break

        # Parse point cloud
        field_info = {}
        for f in msg.fields:
            field_info[f.name] = (f.offset, f.datatype)

        data = np.frombuffer(msg.data, dtype=np.uint8)
        num_points = msg.width * msg.height

        x_offset = field_info.get('x', (0, 7))[0]
        y_offset = field_info.get('y', (4, 7))[0]
        z_offset = field_info.get('z', (8, 7))[0]

        for i in range(min(num_points, 5000)):
            base = i * msg.point_step
            x = struct.unpack('f', data[base+x_offset:base+x_offset+4])[0]
            y = struct.unpack('f', data[base+y_offset:base+y_offset+4])[0]
            z = struct.unpack('f', data[base+z_offset:base+z_offset+4])[0]

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                dist = np.sqrt(x*x + y*y + z*z)
                if 0.5 < dist < 100:  # Valid range
                    all_points.append([x, y, z])

        frame_count += 1

    bag.close()

    if len(all_points) == 0:
        print("No valid points found!")
        return None

    points = np.array(all_points)

    print(f"\nAnalyzed {len(points)} points from {frame_count} frames")

    # Statistics for each axis
    print(f"\nAxis Statistics:")
    print(f"  X: min={points[:,0].min():.2f}, max={points[:,0].max():.2f}, mean={points[:,0].mean():.2f}")
    print(f"  Y: min={points[:,1].min():.2f}, max={points[:,1].max():.2f}, mean={points[:,1].mean():.2f}")
    print(f"  Z: min={points[:,2].min():.2f}, max={points[:,2].max():.2f}, mean={points[:,2].mean():.2f}")

    # Determine forward direction based on point distribution
    x_positive = np.sum(points[:,0] > 0)
    x_negative = np.sum(points[:,0] < 0)
    y_positive = np.sum(points[:,1] > 0)
    y_negative = np.sum(points[:,1] < 0)
    z_positive = np.sum(points[:,2] > 0)
    z_negative = np.sum(points[:,2] < 0)

    total = len(points)
    print(f"\nPoint Distribution:")
    print(f"  X > 0: {x_positive/total*100:.1f}%, X < 0: {x_negative/total*100:.1f}%")
    print(f"  Y > 0: {y_positive/total*100:.1f}%, Y < 0: {y_negative/total*100:.1f}%")
    print(f"  Z > 0: {z_positive/total*100:.1f}%, Z < 0: {z_negative/total*100:.1f}%")

    # Determine likely forward direction
    # For a forward-facing LiDAR, most points should be in front (positive X in standard ROS frame)
    print(f"\nCoordinate Frame Interpretation:")

    if x_negative/total > 0.6:
        print(f"  -> LiDAR X axis points BACKWARD (most points have X < 0)")
        lidar_forward = "-X"
    elif x_positive/total > 0.6:
        print(f"  -> LiDAR X axis points FORWARD (most points have X > 0)")
        lidar_forward = "+X"
    elif y_positive/total > 0.6:
        print(f"  -> LiDAR Y axis points FORWARD (most points have Y > 0)")
        lidar_forward = "+Y"
    elif y_negative/total > 0.6:
        print(f"  -> LiDAR -Y axis points FORWARD (most points have Y < 0)")
        lidar_forward = "-Y"
    else:
        print(f"  -> Cannot determine forward direction clearly")
        lidar_forward = "unclear"

    # Ground plane analysis (Z should be mostly near 0 or negative for ground)
    ground_points = points[np.abs(points[:,2]) < 0.5]
    if len(ground_points) > 100:
        print(f"  -> Ground plane detected at Z ≈ 0 ({len(ground_points)} points)")
        lidar_up = "+Z"
    elif z_negative/total > 0.7:
        print(f"  -> LiDAR Z axis likely points DOWN")
        lidar_up = "-Z"
    else:
        print(f"  -> LiDAR Z axis likely points UP")
        lidar_up = "+Z"

    return {
        'forward': lidar_forward,
        'up': lidar_up,
        'stats': {
            'x_pos_ratio': x_positive/total,
            'y_pos_ratio': y_positive/total,
            'z_pos_ratio': z_positive/total
        }
    }


def analyze_imu_frame(bag_path, topic='/imu/data', num_samples=1000):
    """Analyze IMU coordinate frame orientation."""
    print("\n" + "=" * 60)
    print("IMU Coordinate Frame Analysis")
    print("=" * 60)

    bag = rosbag.Bag(bag_path)

    acc_data = []
    gyro_data = []
    quat_data = []

    count = 0
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if count >= num_samples:
            break

        acc_data.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        gyro_data.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        quat_data.append([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        count += 1

    bag.close()

    if len(acc_data) == 0:
        print("No IMU data found!")
        return None

    acc = np.array(acc_data)
    gyro = np.array(gyro_data)
    quat = np.array(quat_data)

    print(f"\nAnalyzed {len(acc)} IMU samples")

    # Accelerometer analysis (should show gravity)
    acc_mean = np.mean(acc, axis=0)
    acc_norm = np.linalg.norm(acc_mean)

    print(f"\nAccelerometer Mean: [{acc_mean[0]:.4f}, {acc_mean[1]:.4f}, {acc_mean[2]:.4f}] m/s²")
    print(f"Accelerometer Norm: {acc_norm:.4f} m/s² (gravity ≈ 9.81)")

    # Determine gravity direction
    gravity_axis = np.argmax(np.abs(acc_mean))
    axis_names = ['X', 'Y', 'Z']
    gravity_sign = '+' if acc_mean[gravity_axis] > 0 else '-'

    print(f"\nGravity Direction: {gravity_sign}{axis_names[gravity_axis]} axis")
    print(f"  (largest acceleration component: {acc_mean[gravity_axis]:.4f} m/s²)")

    # IMU up direction
    if gravity_axis == 2:  # Z
        if acc_mean[2] > 0:
            imu_up = "+Z"
            print(f"  -> IMU frame: Z-up (gravity = +Z, FLU/ENU style)")
        else:
            imu_up = "-Z"
            print(f"  -> IMU frame: Z-down (gravity = -Z, FRD/NED style)")
    elif gravity_axis == 0:  # X
        imu_up = "+X" if acc_mean[0] > 0 else "-X"
        print(f"  -> IMU frame: unusual (gravity in X axis)")
    else:  # Y
        imu_up = "+Y" if acc_mean[1] > 0 else "-Y"
        print(f"  -> IMU frame: unusual (gravity in Y axis)")

    # Gyroscope analysis
    gyro_mean = np.mean(gyro, axis=0)
    gyro_std = np.std(gyro, axis=0)
    print(f"\nGyroscope Mean: [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}] rad/s")
    print(f"Gyroscope Std:  [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}] rad/s")

    # Initial orientation from quaternion
    r = R.from_quat(quat[0])
    euler = r.as_euler('xyz', degrees=True)
    print(f"\nInitial IMU Orientation (quaternion -> euler):")
    print(f"  Roll:  {euler[0]:.2f}°")
    print(f"  Pitch: {euler[1]:.2f}°")
    print(f"  Yaw:   {euler[2]:.2f}°")

    return {
        'up': imu_up,
        'gravity_axis': axis_names[gravity_axis],
        'gravity_value': acc_mean[gravity_axis],
        'initial_euler': euler
    }


def compute_extrinsic_rotation(lidar_info, imu_info):
    """Compute the required extrinsic rotation matrix."""
    print("\n" + "=" * 60)
    print("Extrinsic Rotation Computation")
    print("=" * 60)

    if lidar_info is None or imu_info is None:
        print("Cannot compute extrinsics - missing data")
        return None

    lidar_forward = lidar_info['forward']
    lidar_up = lidar_info['up']
    imu_up = imu_info['up']

    print(f"\nDetected Configuration:")
    print(f"  LiDAR forward: {lidar_forward}")
    print(f"  LiDAR up: {lidar_up}")
    print(f"  IMU up: {imu_up}")

    # LIO-SAM expects:
    # - X forward
    # - Y left
    # - Z up

    print(f"\nLIO-SAM expects: X-forward, Y-left, Z-up (ROS REP-103)")

    # Determine required rotation
    rotation_matrix = np.eye(3)
    rotation_description = "Identity (no rotation needed)"

    if lidar_forward == "-X":
        # LiDAR X points backward -> need 180° rotation around Z
        print(f"\n  LiDAR X points backward -> Need 180° rotation around Z")
        rotation_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        rotation_description = "180° around Z (Rz_180)"

    elif lidar_forward == "+Y":
        # LiDAR Y points forward -> need -90° rotation around Z
        print(f"\n  LiDAR Y points forward -> Need -90° rotation around Z")
        rotation_matrix = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
        rotation_description = "-90° around Z (Rz_-90)"

    elif lidar_forward == "-Y":
        # LiDAR -Y points forward -> need +90° rotation around Z
        print(f"\n  LiDAR -Y points forward -> Need +90° rotation around Z")
        rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        rotation_description = "+90° around Z (Rz_90)"

    elif lidar_forward == "+X":
        print(f"\n  LiDAR X already points forward -> No rotation needed")

    print(f"\nRecommended Rotation: {rotation_description}")
    print(f"\nRotation Matrix (extrinsicRot):")
    print(f"  [{rotation_matrix[0,0]}, {rotation_matrix[0,1]}, {rotation_matrix[0,2]},")
    print(f"   {rotation_matrix[1,0]}, {rotation_matrix[1,1]}, {rotation_matrix[1,2]},")
    print(f"   {rotation_matrix[2,0]}, {rotation_matrix[2,1]}, {rotation_matrix[2,2]}]")

    # extrinsicRPY is typically the same as extrinsicRot for rotation-only transforms
    print(f"\nRotation Matrix (extrinsicRPY) - same as extrinsicRot:")
    print(f"  [{rotation_matrix[0,0]}, {rotation_matrix[0,1]}, {rotation_matrix[0,2]},")
    print(f"   {rotation_matrix[1,0]}, {rotation_matrix[1,1]}, {rotation_matrix[1,2]},")
    print(f"   {rotation_matrix[2,0]}, {rotation_matrix[2,1]}, {rotation_matrix[2,2]}]")

    return rotation_matrix


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_extrinsics.py <bag_file> [lidar_topic] [imu_topic]")
        sys.exit(1)

    bag_path = sys.argv[1]
    lidar_topic = sys.argv[2] if len(sys.argv) > 2 else '/lidar_points'
    imu_topic = sys.argv[3] if len(sys.argv) > 3 else '/imu/data'

    print("=" * 60)
    print("LIO-SAM Extrinsics Analysis Tool")
    print("=" * 60)
    print(f"Bag file: {bag_path}")
    print(f"LiDAR topic: {lidar_topic}")
    print(f"IMU topic: {imu_topic}")

    # Analyze LiDAR frame
    lidar_info = analyze_lidar_frame(bag_path, lidar_topic)

    # Analyze IMU frame
    imu_info = analyze_imu_frame(bag_path, imu_topic)

    # Compute extrinsic rotation
    rotation = compute_extrinsic_rotation(lidar_info, imu_info)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("\nUpdate your params.yaml with the recommended extrinsic matrices above.")
    print("Remember to rebuild after changing params.yaml (if params are compiled in).")


if __name__ == '__main__':
    main()
