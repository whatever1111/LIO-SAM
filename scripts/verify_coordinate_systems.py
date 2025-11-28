#!/usr/bin/env python3
"""
Coordinate System Verification Script for LIO-SAM
This script analyzes the IMU, LiDAR, and GPS data to verify coordinate system consistency.

Expected Coordinate Systems:
- ROS REP-105: X-forward, Y-left, Z-up (FLU / ENU)
- IMU: Should output acceleration with Z ≈ +9.8 when stationary (gravity pointing up)
- LiDAR: Points should be in the same frame as IMU
- GPS: FPA odometry is in ECEF, needs conversion to local ENU
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct

def analyze_imu_data(bag_path, topic='/imu/data', num_samples=1000):
    """Analyze IMU data to verify coordinate system"""
    print("\n" + "="*60)
    print("IMU Data Analysis")
    print("="*60)

    accel_data = []
    gyro_data = []
    orientation_data = []
    timestamps = []

    bag = rosbag.Bag(bag_path, 'r')
    count = 0

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if count >= num_samples:
            break

        accel_data.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        gyro_data.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        orientation_data.append([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        timestamps.append(t.to_sec())
        count += 1

    bag.close()

    accel_data = np.array(accel_data)
    gyro_data = np.array(gyro_data)
    orientation_data = np.array(orientation_data)

    print(f"\nSamples analyzed: {count}")
    print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} ({timestamps[-1]-timestamps[0]:.2f}s)")

    # Analyze acceleration (should show gravity)
    print("\n--- Acceleration Analysis (m/s²) ---")
    print(f"Mean: X={accel_data[:,0].mean():.4f}, Y={accel_data[:,1].mean():.4f}, Z={accel_data[:,2].mean():.4f}")
    print(f"Std:  X={accel_data[:,0].std():.4f}, Y={accel_data[:,1].std():.4f}, Z={accel_data[:,2].std():.4f}")

    accel_magnitude = np.linalg.norm(accel_data, axis=1)
    print(f"Magnitude: mean={accel_magnitude.mean():.4f}, std={accel_magnitude.std():.4f}")

    # Determine gravity direction
    mean_accel = accel_data.mean(axis=0)
    gravity_magnitude = np.linalg.norm(mean_accel)
    gravity_direction = mean_accel / gravity_magnitude
    print(f"\nGravity direction (normalized): [{gravity_direction[0]:.4f}, {gravity_direction[1]:.4f}, {gravity_direction[2]:.4f}]")

    # Check if gravity is pointing up (+Z in FLU convention)
    if gravity_direction[2] > 0.9:
        print("✓ Gravity is pointing UP (+Z) - Correct for FLU/ENU convention (IMU pointing up)")
    elif gravity_direction[2] < -0.9:
        print("✓ Gravity is pointing DOWN (-Z) - IMU is mounted with Z-down")
        print("  Note: For LIO-SAM, the measured acceleration should be ≈ +9.8 on Z when stationary")
    else:
        print(f"⚠ Gravity is NOT aligned with Z-axis!")
        print(f"  This suggests IMU is tilted or coordinate system is different")

    # Analyze gyroscope
    print("\n--- Gyroscope Analysis (rad/s) ---")
    print(f"Mean: X={gyro_data[:,0].mean():.6f}, Y={gyro_data[:,1].mean():.6f}, Z={gyro_data[:,2].mean():.6f}")
    print(f"Std:  X={gyro_data[:,0].std():.6f}, Y={gyro_data[:,1].std():.6f}, Z={gyro_data[:,2].std():.6f}")

    # Analyze orientation (quaternion)
    print("\n--- Orientation Analysis (quaternion) ---")
    print(f"Mean: x={orientation_data[:,0].mean():.4f}, y={orientation_data[:,1].mean():.4f}, z={orientation_data[:,2].mean():.4f}, w={orientation_data[:,3].mean():.4f}")

    # Convert first few quaternions to roll-pitch-yaw
    print("\n--- First 5 orientations as RPY (degrees) ---")
    for i in range(min(5, len(orientation_data))):
        q = orientation_data[i]
        r = R.from_quat(q)  # scipy uses [x,y,z,w] format
        rpy = r.as_euler('xyz', degrees=True)
        print(f"  {i}: Roll={rpy[0]:.2f}, Pitch={rpy[1]:.2f}, Yaw={rpy[2]:.2f}")

    return accel_data, gyro_data, orientation_data


def analyze_lidar_data(bag_path, topic='/lidar_points', num_frames=10):
    """Analyze LiDAR point cloud data"""
    print("\n" + "="*60)
    print("LiDAR Data Analysis")
    print("="*60)

    bag = rosbag.Bag(bag_path, 'r')
    count = 0

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if count >= num_frames:
            break

        print(f"\n--- Frame {count} ---")
        print(f"Timestamp: {t.to_sec():.4f}")
        print(f"Frame ID: {msg.header.frame_id}")
        print(f"Dimensions: height={msg.height}, width={msg.width}")
        print(f"Point step: {msg.point_step} bytes")
        print(f"Row step: {msg.row_step} bytes")
        print(f"Is dense: {msg.is_dense}")

        # Parse fields
        print(f"Fields:")
        field_offsets = {}
        for field in msg.fields:
            print(f"  {field.name}: offset={field.offset}, datatype={field.datatype}, count={field.count}")
            field_offsets[field.name] = field.offset

        # Extract some points to analyze coordinate range
        num_points = msg.width * msg.height
        print(f"Total points: {num_points}")

        if num_points > 0:
            # Parse first 100 points
            points_x = []
            points_y = []
            points_z = []

            for i in range(min(1000, num_points)):
                offset = i * msg.point_step
                x = struct.unpack('f', msg.data[offset:offset+4])[0]
                y = struct.unpack('f', msg.data[offset+4:offset+8])[0]
                z = struct.unpack('f', msg.data[offset+8:offset+12])[0]

                if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                    points_x.append(x)
                    points_y.append(y)
                    points_z.append(z)

            if points_x:
                points_x = np.array(points_x)
                points_y = np.array(points_y)
                points_z = np.array(points_z)

                print(f"\n  Point coordinate ranges (sample of {len(points_x)} points):")
                print(f"    X: [{points_x.min():.2f}, {points_x.max():.2f}], mean={points_x.mean():.2f}")
                print(f"    Y: [{points_y.min():.2f}, {points_y.max():.2f}], mean={points_y.mean():.2f}")
                print(f"    Z: [{points_z.min():.2f}, {points_z.max():.2f}], mean={points_z.mean():.2f}")

                distances = np.sqrt(points_x**2 + points_y**2 + points_z**2)
                print(f"    Distance: [{distances.min():.2f}, {distances.max():.2f}], mean={distances.mean():.2f}")

        count += 1

    bag.close()


def analyze_fpa_odometry(bag_path, topic='/fixposition/fpa/odometry', num_samples=100):
    """Analyze FPA odometry data (ECEF)"""
    print("\n" + "="*60)
    print("FPA Odometry Analysis (ECEF)")
    print("="*60)

    bag = rosbag.Bag(bag_path, 'r')
    positions = []
    orientations = []
    timestamps = []
    count = 0

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if count >= num_samples:
            break

        positions.append([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        orientations.append([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        timestamps.append(t.to_sec())
        count += 1

    bag.close()

    positions = np.array(positions)
    orientations = np.array(orientations)

    print(f"\nSamples analyzed: {count}")

    # ECEF position analysis
    print("\n--- ECEF Position Analysis ---")
    print(f"X range: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    print(f"Y range: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
    print(f"Z range: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

    # First position (origin)
    origin_ecef = positions[0]
    print(f"\nFirst position (ECEF): [{origin_ecef[0]:.2f}, {origin_ecef[1]:.2f}, {origin_ecef[2]:.2f}]")

    # Convert ECEF to LLA
    lat, lon, alt = ecef_to_lla(origin_ecef[0], origin_ecef[1], origin_ecef[2])
    print(f"First position (LLA): lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")

    # Convert to local ENU
    R_ecef_to_enu = get_ecef_to_enu_matrix(lat, lon)

    enu_positions = []
    for pos in positions:
        delta = pos - origin_ecef
        enu = R_ecef_to_enu @ delta
        enu_positions.append(enu)

    enu_positions = np.array(enu_positions)

    print("\n--- Local ENU Position Analysis ---")
    print(f"E (East) range: [{enu_positions[:,0].min():.2f}, {enu_positions[:,0].max():.2f}]m")
    print(f"N (North) range: [{enu_positions[:,1].min():.2f}, {enu_positions[:,1].max():.2f}]m")
    print(f"U (Up) range: [{enu_positions[:,2].min():.2f}, {enu_positions[:,2].max():.2f}]m")

    # Orientation analysis
    print("\n--- Orientation Analysis ---")
    print(f"First orientation (quaternion): x={orientations[0,0]:.4f}, y={orientations[0,1]:.4f}, z={orientations[0,2]:.4f}, w={orientations[0,3]:.4f}")

    # Convert to RPY
    for i in range(min(5, len(orientations))):
        q = orientations[i]
        r = R.from_quat(q)
        rpy = r.as_euler('xyz', degrees=True)
        print(f"  {i}: Roll={rpy[0]:.2f}°, Pitch={rpy[1]:.2f}°, Yaw={rpy[2]:.2f}°")

    return positions, enu_positions, orientations


def ecef_to_lla(x, y, z):
    """Convert ECEF to latitude, longitude, altitude (WGS84)"""
    a = 6378137.0  # semi-major axis
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - e2))

    # Iterative refinement
    for _ in range(5):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - e2 * N / (N + alt)))

    return np.degrees(lat), np.degrees(lon), alt


def get_ecef_to_enu_matrix(lat_deg, lon_deg):
    """Get rotation matrix from ECEF to local ENU"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R = np.array([
        [-sin_lon,           cos_lon,          0],
        [-sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat],
        [ cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])

    return R


def check_coordinate_consistency(imu_accel, enu_positions):
    """Check if IMU and GPS coordinate systems are consistent"""
    print("\n" + "="*60)
    print("Coordinate System Consistency Check")
    print("="*60)

    # IMU gravity should be approximately [0, 0, +9.8] for FLU convention
    mean_accel = imu_accel.mean(axis=0)
    print(f"\nIMU mean acceleration: [{mean_accel[0]:.4f}, {mean_accel[1]:.4f}, {mean_accel[2]:.4f}] m/s²")

    # Check gravity direction
    gravity_magnitude = np.linalg.norm(mean_accel)
    if abs(gravity_magnitude - 9.8) > 1.0:
        print(f"⚠ Gravity magnitude {gravity_magnitude:.2f} is not close to 9.8 m/s²")
    else:
        print(f"✓ Gravity magnitude {gravity_magnitude:.2f} is close to expected 9.8 m/s²")

    # GPS movement analysis
    if len(enu_positions) > 1:
        displacement = enu_positions[-1] - enu_positions[0]
        total_distance = np.linalg.norm(displacement[:2])  # horizontal distance
        print(f"\nGPS total horizontal displacement: {total_distance:.2f}m")
        print(f"GPS displacement vector (ENU): E={displacement[0]:.2f}m, N={displacement[1]:.2f}m, U={displacement[2]:.2f}m")

        # Direction of movement
        if total_distance > 1.0:
            direction = np.arctan2(displacement[0], displacement[1])  # angle from North
            print(f"Movement direction: {np.degrees(direction):.1f}° from North (clockwise)")


def verify_extrinsics():
    """Verify extrinsic parameters from params.yaml"""
    print("\n" + "="*60)
    print("Extrinsic Parameters Analysis")
    print("="*60)

    # Current configuration from params.yaml
    extrinsicTrans = [0.0, 0.0, 0.0]
    extrinsicRot = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    extrinsicRPY = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    print("\nCurrent configuration:")
    print(f"  extrinsicTrans (LiDAR -> IMU): {extrinsicTrans}")
    print(f"  extrinsicRot (identity): {extrinsicRot.flatten().tolist()}")
    print(f"  extrinsicRPY (identity): {extrinsicRPY.flatten().tolist()}")

    print("\nInterpretation:")
    print("  - LiDAR and IMU are at the same position (zero translation)")
    print("  - LiDAR and IMU have the same orientation (identity rotation)")
    print("  - This means IMU data will pass through unchanged")

    print("\nLIO-SAM IMU Conversion:")
    print("  - extQRPY = Quaternion(extrinsicRPY).inverse()")
    print("  - With identity matrix, extQRPY = identity quaternion")
    print("  - IMU orientation: q_out = q_in * extQRPY = q_in (unchanged)")
    print("  - IMU acceleration: acc_out = extRot * acc_in = acc_in (unchanged)")


def main():
    bag_path = "/root/autodl-tmp/info_fixed.bag"

    print("="*60)
    print("LIO-SAM Coordinate System Verification")
    print("="*60)
    print(f"Bag file: {bag_path}")

    # Verify extrinsic parameters
    verify_extrinsics()

    # Analyze IMU data
    imu_accel, imu_gyro, imu_orient = analyze_imu_data(bag_path)

    # Analyze LiDAR data
    analyze_lidar_data(bag_path)

    # Analyze FPA odometry
    ecef_pos, enu_pos, fpa_orient = analyze_fpa_odometry(bag_path)

    # Check consistency
    check_coordinate_consistency(imu_accel, enu_pos)

    print("\n" + "="*60)
    print("Summary and Recommendations")
    print("="*60)

    # Summary based on analysis
    mean_accel = imu_accel.mean(axis=0)
    gravity_z = mean_accel[2]

    print("\nCoordinate System Status:")

    if gravity_z > 8.0:
        print("✓ IMU Z-axis points UP (positive gravity on Z)")
        print("  This is correct for FLU/ENU convention with Z-up")
    elif gravity_z < -8.0:
        print("⚠ IMU Z-axis points DOWN (negative gravity on Z)")
        print("  The extrinsic rotation may need adjustment")
        print("  Consider: extrinsicRot = diag(1, 1, -1) or similar")
    else:
        print("⚠ IMU has unusual gravity reading")
        print("  Check IMU calibration or extrinsic parameters")

    print("\nFor LIO-SAM to work correctly:")
    print("1. IMU acceleration Z should be approximately +9.8 when stationary")
    print("2. IMU orientation should report 0° roll/pitch when level")
    print("3. GPS/ENU coordinates should be in East-North-Up format")
    print("4. LiDAR frame should match IMU frame (via extrinsics)")


if __name__ == "__main__":
    main()
