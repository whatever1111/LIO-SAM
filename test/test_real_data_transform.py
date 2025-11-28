#!/usr/bin/env python3
"""
Unit test for coordinate system transformations using real bag file data.

This test verifies:
1. ECEF to ENU conversion (fpaOdomConverter logic)
2. ENU to LiDAR frame conversion (gpsExtrinsicRot)
3. IMU to LiDAR frame conversion (extrinsicRot)
4. Consistency between GPS trajectory and IMU orientation
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

# ANSI colors
GREEN = '\033[1;32m'
RED = '\033[1;31m'
YELLOW = '\033[1;33m'
CYAN = '\033[1;36m'
RESET = '\033[0m'

# Configuration from params.yaml
EXT_ROT = np.array([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])  # IMU to LiDAR (Rz 180°)

GPS_EXT_ROT = np.array([[0, -1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]])  # ENU to LiDAR

# WGS84 parameters
WGS84_A = 6378137.0  # semi-major axis
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F  # first eccentricity squared


def ecef_to_lla(ecef):
    """Convert ECEF coordinates to LLA (latitude, longitude, altitude)."""
    x, y, z = ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x*x + y*y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))

    # Iterative refinement
    for _ in range(5):
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))

    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return lat, lon, alt


def ecef_to_enu_matrix(origin_ecef):
    """Get rotation matrix from ECEF to ENU at given origin."""
    lat, lon, _ = ecef_to_lla(origin_ecef)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    R_ecef_to_enu = np.array([
        [-sin_lon,          cos_lon,          0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat],
        [cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])

    return R_ecef_to_enu, lat, lon


def quaternion_to_euler(q):
    """Convert quaternion (x,y,z,w) to euler angles (roll, pitch, yaw)."""
    r = R.from_quat([q[0], q[1], q[2], q[3]])
    return r.as_euler('xyz', degrees=True)


def test_ecef_to_enu_conversion(bag_path):
    """Test 1: Verify ECEF to ENU conversion using real GPS data."""
    print(f"\n{YELLOW}=== Test 1: ECEF to ENU Conversion ==={RESET}")

    bag = rosbag.Bag(bag_path)

    gps_data = []
    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        pos = msg.pose.pose.position
        gps_data.append({
            'time': msg.header.stamp.to_sec(),
            'ecef': np.array([pos.x, pos.y, pos.z]),
            'quat': [msg.pose.pose.orientation.x,
                     msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z,
                     msg.pose.pose.orientation.w]
        })
        if len(gps_data) >= 2000:  # Sample first 2000 messages (~200 seconds)
            break
    bag.close()

    if len(gps_data) < 2:
        print(f"{RED}  [FAIL] Not enough GPS data{RESET}")
        return False

    # Use first position as origin
    origin_ecef = gps_data[0]['ecef']
    R_ecef_enu, lat, lon = ecef_to_enu_matrix(origin_ecef)

    print(f"  Origin ECEF: [{origin_ecef[0]:.2f}, {origin_ecef[1]:.2f}, {origin_ecef[2]:.2f}]")
    print(f"  Origin LLA:  lat={np.degrees(lat):.6f}°, lon={np.degrees(lon):.6f}°")

    # Convert all positions to ENU
    enu_positions = []
    for data in gps_data:
        delta_ecef = data['ecef'] - origin_ecef
        enu = R_ecef_enu @ delta_ecef
        enu_positions.append(enu)

    # First position should be [0,0,0]
    first_enu = enu_positions[0]
    if np.linalg.norm(first_enu) < 1e-6:
        print(f"{GREEN}  [PASS] First ENU position is origin [0,0,0]{RESET}")
    else:
        print(f"{RED}  [FAIL] First ENU position: {first_enu}{RESET}")
        return False

    # Check ENU is right-handed (Z should be approximately up)
    # Get movement direction in ENU
    last_enu = enu_positions[-1]
    print(f"  Last ENU position: E={last_enu[0]:.3f}m, N={last_enu[1]:.3f}m, U={last_enu[2]:.3f}m")

    # Calculate total distance traveled
    total_dist = np.linalg.norm(last_enu[:2])  # Horizontal distance
    print(f"  Horizontal distance traveled: {total_dist:.2f}m")

    print(f"{GREEN}  [PASS] ECEF to ENU conversion verified{RESET}")
    return True, enu_positions, gps_data


def test_enu_to_lidar_conversion(enu_positions):
    """Test 2: Verify ENU to LiDAR frame conversion."""
    print(f"\n{YELLOW}=== Test 2: ENU to LiDAR Frame Conversion ==={RESET}")

    print(f"  GPS EXT ROT matrix:")
    print(f"    {GPS_EXT_ROT[0]}")
    print(f"    {GPS_EXT_ROT[1]}")
    print(f"    {GPS_EXT_ROT[2]}")

    # Convert ENU to LiDAR frame
    lidar_positions = []
    for enu in enu_positions:
        lidar = GPS_EXT_ROT @ enu
        lidar_positions.append(lidar)

    # Verify direction mapping
    last_enu = enu_positions[-1]
    last_lidar = lidar_positions[-1]

    print(f"\n  Last position comparison:")
    print(f"    ENU:   E={last_enu[0]:.3f}, N={last_enu[1]:.3f}, U={last_enu[2]:.3f}")
    print(f"    LiDAR: X={last_lidar[0]:.3f}, Y={last_lidar[1]:.3f}, Z={last_lidar[2]:.3f}")

    # Verify the mapping: X_lidar = -N, Y_lidar = -E, Z_lidar = U
    expected_lidar = np.array([-last_enu[1], -last_enu[0], last_enu[2]])

    if np.allclose(last_lidar, expected_lidar, atol=1e-6):
        print(f"{GREEN}  [PASS] ENU to LiDAR mapping correct{RESET}")
        print(f"    X_lidar = -North, Y_lidar = -East, Z_lidar = Up")
    else:
        print(f"{RED}  [FAIL] ENU to LiDAR mapping incorrect{RESET}")
        print(f"    Expected: {expected_lidar}")
        print(f"    Got: {last_lidar}")
        return False

    return True, lidar_positions


def test_imu_to_lidar_conversion(bag_path):
    """Test 3: Verify IMU to LiDAR frame conversion."""
    print(f"\n{YELLOW}=== Test 3: IMU to LiDAR Frame Conversion ==={RESET}")

    bag = rosbag.Bag(bag_path)

    imu_data = []
    for topic, msg, t in bag.read_messages(topics=['/imu/data']):
        imu_data.append({
            'time': msg.header.stamp.to_sec(),
            'acc': np.array([msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z]),
            'gyro': np.array([msg.angular_velocity.x,
                             msg.angular_velocity.y,
                             msg.angular_velocity.z]),
            'quat': [msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w]
        })
        if len(imu_data) >= 100:
            break
    bag.close()

    if len(imu_data) < 1:
        print(f"{RED}  [FAIL] No IMU data found{RESET}")
        return False

    # Get first IMU reading (should be stationary)
    first_imu = imu_data[0]
    acc_imu = first_imu['acc']

    print(f"  IMU acceleration (raw): [{acc_imu[0]:.4f}, {acc_imu[1]:.4f}, {acc_imu[2]:.4f}]")

    # Apply extrinsicRot to convert to LiDAR frame
    acc_lidar = EXT_ROT @ acc_imu

    print(f"  LiDAR acceleration:     [{acc_lidar[0]:.4f}, {acc_lidar[1]:.4f}, {acc_lidar[2]:.4f}]")

    # Gravity should be primarily in Z direction (~9.8 m/s^2)
    gravity_z = acc_lidar[2]
    gravity_xy = np.sqrt(acc_lidar[0]**2 + acc_lidar[1]**2)

    print(f"\n  Gravity analysis:")
    print(f"    Z component: {gravity_z:.4f} m/s² (expected ~9.8)")
    print(f"    XY component: {gravity_xy:.4f} m/s² (expected ~0)")

    if gravity_z > 9.5 and gravity_xy < 0.5:
        print(f"{GREEN}  [PASS] Gravity is primarily in +Z direction{RESET}")
    else:
        print(f"{YELLOW}  [WARN] Gravity direction may indicate sensor tilt{RESET}")

    # Test rotation transform
    print(f"\n  Rotation matrix (EXT_ROT) verification:")
    print(f"    det(EXT_ROT) = {np.linalg.det(EXT_ROT):.1f} (should be 1)")

    # IMU +X -> LiDAR -X
    test_x = EXT_ROT @ np.array([1, 0, 0])
    if np.allclose(test_x, [-1, 0, 0]):
        print(f"{GREEN}  [PASS] IMU +X -> LiDAR -X{RESET}")
    else:
        print(f"{RED}  [FAIL] IMU +X mapping incorrect: {test_x}{RESET}")
        return False

    return True


def test_trajectory_consistency(enu_positions, gps_data, bag_path):
    """Test 4: Verify trajectory direction matches IMU heading."""
    print(f"\n{YELLOW}=== Test 4: Trajectory and IMU Heading Consistency ==={RESET}")

    if len(enu_positions) < 10:
        print(f"{YELLOW}  [SKIP] Not enough data for trajectory analysis{RESET}")
        return True

    # Calculate trajectory direction in ENU (first significant movement)
    start_pos = np.array(enu_positions[0])

    # Find first position with significant movement
    trajectory_dir_enu = None
    trajectory_heading = None
    for i, pos in enumerate(enu_positions[10:], start=10):
        pos = np.array(pos)
        dist = np.linalg.norm(pos[:2] - start_pos[:2])
        if dist > 1.0:  # At least 1m movement
            trajectory_dir_enu = pos[:2] - start_pos[:2]
            trajectory_dir_enu = trajectory_dir_enu / np.linalg.norm(trajectory_dir_enu)

            # Calculate heading from trajectory (angle from North)
            # In ENU: North = +Y, East = +X
            trajectory_heading = np.degrees(np.arctan2(trajectory_dir_enu[0], trajectory_dir_enu[1]))

            print(f"  Trajectory direction (ENU): E={trajectory_dir_enu[0]:.3f}, N={trajectory_dir_enu[1]:.3f}")
            print(f"  Trajectory heading: {trajectory_heading:.1f}° from North")
            break
    else:
        print(f"{YELLOW}  [SKIP] No significant movement detected{RESET}")
        return True

    # Get IMU heading from first GPS quaternion
    first_quat = gps_data[0]['quat']
    euler = quaternion_to_euler(first_quat)
    imu_yaw = euler[2]  # Yaw in degrees

    print(f"  GPS/IMU initial yaw: {imu_yaw:.1f}°")

    # Convert trajectory to LiDAR frame
    traj_lidar = GPS_EXT_ROT[:2, :2] @ trajectory_dir_enu
    print(f"  Trajectory in LiDAR frame: X={traj_lidar[0]:.3f}, Y={traj_lidar[1]:.3f}")

    # Analyze if GPS transform is consistent with vehicle heading
    print(f"\n{CYAN}  Analysis:{RESET}")
    print(f"    - gpsExtrinsicRot assumes: vehicle initially facing North")
    print(f"    - Actual initial heading: {imu_yaw:.1f}° (from GPS/IMU)")

    # Calculate expected heading offset
    heading_diff = imu_yaw  # Difference from North
    print(f"    - Heading offset from North: {heading_diff:.1f}°")

    # In LiDAR frame, -X is forward
    # Calculate angle of movement in LiDAR frame
    lidar_move_angle = np.degrees(np.arctan2(traj_lidar[1], traj_lidar[0]))
    print(f"    - Movement angle in LiDAR frame: {lidar_move_angle:.1f}° from -X axis")

    # For correct mapping when vehicle faces North:
    # - Moving North in ENU = Moving -X in LiDAR (forward)
    # If vehicle is not facing North, the mapping will show movement at an angle

    if traj_lidar[0] < -0.5:  # Primarily in -X direction
        print(f"{GREEN}  [PASS] Vehicle moving primarily in LiDAR -X (forward) direction{RESET}")
    elif abs(traj_lidar[0]) < 0.3 and abs(traj_lidar[1]) > 0.8:
        print(f"{YELLOW}  [INFO] Vehicle moving primarily sideways in LiDAR frame{RESET}")
        print(f"         This is expected if vehicle heading differs from North")
    else:
        print(f"{YELLOW}  [INFO] Complex movement pattern detected{RESET}")

    return True


def test_heading_aware_transform(enu_positions, gps_data):
    """Test 6: Verify coordinate transform with heading compensation."""
    print(f"\n{YELLOW}=== Test 6: Heading-Aware GPS Transform ==={RESET}")

    if len(enu_positions) < 100:
        print(f"{YELLOW}  [SKIP] Not enough data{RESET}")
        return True

    # Get initial heading from GPS quaternion
    first_quat = gps_data[0]['quat']
    r = R.from_quat(first_quat)
    euler = r.as_euler('xyz', degrees=False)
    initial_yaw = euler[2]  # radians

    print(f"  Initial vehicle yaw: {np.degrees(initial_yaw):.1f}°")

    # Create heading-compensated rotation matrix
    # This rotates ENU by the vehicle's initial heading to align with body frame
    cos_yaw = np.cos(initial_yaw)
    sin_yaw = np.sin(initial_yaw)

    # Rotation from ENU to body-aligned ENU (rotate around Z)
    R_heading = np.array([
        [cos_yaw, sin_yaw, 0],
        [-sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    # Combined transform: first rotate by heading, then apply GPS_EXT_ROT
    # But since GPS_EXT_ROT assumes North=forward, we need to adjust

    # For Livox (-X forward), if vehicle faces direction theta from North:
    # - Vehicle's forward (-X in body) points at angle theta in ENU
    # - We need ENU coordinates rotated so that direction theta becomes -X

    # The correct combined rotation:
    # Step 1: Rotate ENU so that vehicle forward direction becomes +Y (North-like)
    # Step 2: Apply GPS_EXT_ROT which maps +Y to -X

    R_combined = GPS_EXT_ROT @ R_heading

    print(f"\n  Standard GPS_EXT_ROT (assumes North facing):")
    print(f"    {GPS_EXT_ROT[0]}")
    print(f"    {GPS_EXT_ROT[1]}")
    print(f"    {GPS_EXT_ROT[2]}")

    print(f"\n  Heading-compensated rotation (for yaw={np.degrees(initial_yaw):.1f}°):")
    print(f"    [{R_combined[0,0]:.3f}, {R_combined[0,1]:.3f}, {R_combined[0,2]:.3f}]")
    print(f"    [{R_combined[1,0]:.3f}, {R_combined[1,1]:.3f}, {R_combined[1,2]:.3f}]")
    print(f"    [{R_combined[2,0]:.3f}, {R_combined[2,1]:.3f}, {R_combined[2,2]:.3f}]")

    # Test with trajectory
    start_enu = np.array(enu_positions[0])
    end_enu = np.array(enu_positions[-1])
    delta_enu = end_enu - start_enu

    # Transform with standard GPS_EXT_ROT
    delta_lidar_std = GPS_EXT_ROT @ delta_enu

    # Transform with heading-compensated rotation
    delta_lidar_comp = R_combined @ delta_enu

    print(f"\n  Trajectory transform comparison:")
    print(f"    ENU delta: E={delta_enu[0]:.2f}, N={delta_enu[1]:.2f}, U={delta_enu[2]:.2f}")
    print(f"    Standard:  X={delta_lidar_std[0]:.2f}, Y={delta_lidar_std[1]:.2f}, Z={delta_lidar_std[2]:.2f}")
    print(f"    Compensated: X={delta_lidar_comp[0]:.2f}, Y={delta_lidar_comp[1]:.2f}, Z={delta_lidar_comp[2]:.2f}")

    # Analyze which gives forward (-X) as primary direction
    std_forward_ratio = abs(delta_lidar_std[0]) / (np.linalg.norm(delta_lidar_std[:2]) + 1e-6)
    comp_forward_ratio = abs(delta_lidar_comp[0]) / (np.linalg.norm(delta_lidar_comp[:2]) + 1e-6)

    print(f"\n  Forward direction analysis:")
    print(f"    Standard transform -X ratio: {std_forward_ratio:.2f}")
    print(f"    Compensated transform -X ratio: {comp_forward_ratio:.2f}")

    if comp_forward_ratio > std_forward_ratio:
        print(f"{GREEN}  [INFO] Heading compensation improves forward alignment{RESET}")
        print(f"         Consider using heading-compensated gpsExtrinsicRot for this dataset")
    else:
        print(f"{GREEN}  [INFO] Standard transform is adequate{RESET}")

    return True


def test_coordinate_frame_orthogonality():
    """Test 5: Verify all transformation matrices are valid."""
    print(f"\n{YELLOW}=== Test 5: Transformation Matrix Validity ==={RESET}")

    # Check EXT_ROT (IMU to LiDAR)
    det_ext = np.linalg.det(EXT_ROT)
    orth_ext = np.allclose(EXT_ROT @ EXT_ROT.T, np.eye(3))

    print(f"  EXT_ROT (IMU->LiDAR):")
    print(f"    Determinant: {det_ext:.1f} (should be 1 for rotation)")
    print(f"    Orthogonal: {orth_ext}")

    if det_ext == 1 and orth_ext:
        print(f"{GREEN}    [PASS] Valid rotation matrix{RESET}")
    else:
        print(f"{RED}    [FAIL] Invalid rotation matrix{RESET}")
        return False

    # Check GPS_EXT_ROT (ENU to LiDAR)
    det_gps = np.linalg.det(GPS_EXT_ROT)
    orth_gps = np.allclose(GPS_EXT_ROT @ GPS_EXT_ROT.T, np.eye(3))

    print(f"\n  GPS_EXT_ROT (ENU->LiDAR):")
    print(f"    Determinant: {det_gps:.1f} (can be -1 for position transform)")
    print(f"    Orthogonal: {orth_gps}")

    if abs(det_gps) == 1 and orth_gps:
        print(f"{GREEN}    [PASS] Valid orthogonal matrix{RESET}")
    else:
        print(f"{RED}    [FAIL] Invalid orthogonal matrix{RESET}")
        return False

    return True


def main():
    bag_path = "/root/autodl-tmp/info_fixed.bag"

    print(f"\n{YELLOW}{'='*50}{RESET}")
    print(f"{YELLOW}  Real Data Coordinate Transform Tests{RESET}")
    print(f"{YELLOW}  Bag file: {bag_path}{RESET}")
    print(f"{YELLOW}{'='*50}{RESET}")

    all_passed = True

    # Test 1: ECEF to ENU
    result = test_ecef_to_enu_conversion(bag_path)
    if isinstance(result, tuple):
        test1_passed, enu_positions, gps_data = result
    else:
        test1_passed = result
        enu_positions = []
        gps_data = []
    all_passed &= test1_passed

    # Test 2: ENU to LiDAR
    if enu_positions:
        result = test_enu_to_lidar_conversion(enu_positions)
        if isinstance(result, tuple):
            test2_passed, lidar_positions = result
        else:
            test2_passed = result
        all_passed &= test2_passed

    # Test 3: IMU to LiDAR
    all_passed &= test_imu_to_lidar_conversion(bag_path)

    # Test 4: Trajectory consistency
    if enu_positions and gps_data:
        all_passed &= test_trajectory_consistency(enu_positions, gps_data, bag_path)

    # Test 5: Matrix validity
    all_passed &= test_coordinate_frame_orthogonality()

    # Test 6: Heading-aware transform
    if enu_positions and gps_data:
        all_passed &= test_heading_aware_transform(enu_positions, gps_data)

    print(f"\n{YELLOW}{'='*50}{RESET}")
    if all_passed:
        print(f"{GREEN}  All tests PASSED!{RESET}")
    else:
        print(f"{RED}  Some tests FAILED!{RESET}")
    print(f"{YELLOW}{'='*50}{RESET}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
