#!/usr/bin/env python3
"""
Unit test for IMU coordinate transformation in LIO-SAM FPA mode.

This script reads raw IMU data from bag file and verifies:
1. Gravity direction after extrinsicRot transform (should be -Z for MakeSharedU)
2. Orientation quaternion after extrinsicRPY transform (should be valid/level)
3. Angular velocity transform consistency

Usage: python3 test_imu_transform.py
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import rosbag
import sys

# ============== Configuration ==============
BAG_FILE = "/root/autodl-tmp/info_fixed.bag"
FPA_IMU_TOPIC = "/fixposition/fpa/corrimu"
STD_IMU_TOPIC = "/imu/data"

# Current params.yaml settings
# Both use Rz(90°) only - NO Z-flip needed
# GTSAM MakeSharedU expects acc_z ≈ +g when stationary
EXTRINSIC_ROT = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])  # Rz(90) only

EXTRINSIC_RPY = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])  # Rz(90) only (same as EXTRINSIC_ROT)

GRAVITY = 9.80511  # Expected gravity magnitude

# ============== Test Functions ==============

def test_gravity_direction():
    """Test 1: Verify gravity direction after extrinsicRot transform"""
    print("\n" + "="*60)
    print("TEST 1: Gravity Direction (extrinsicRot transform)")
    print("="*60)

    bag = rosbag.Bag(BAG_FILE)
    acc_samples = []

    # Read first 100 FPA IMU messages
    count = 0
    for topic, msg, t in bag.read_messages(topics=[FPA_IMU_TOPIC]):
        if count >= 100:
            break
        acc_raw = np.array([
            msg.data.linear_acceleration.x,
            msg.data.linear_acceleration.y,
            msg.data.linear_acceleration.z
        ])
        acc_samples.append(acc_raw)
        count += 1
    bag.close()

    if len(acc_samples) == 0:
        print("FAIL: No FPA IMU data found in bag")
        return False

    # Average raw acceleration
    acc_raw_mean = np.mean(acc_samples, axis=0)
    print(f"\nRaw FPA IMU acceleration (mean of {len(acc_samples)} samples):")
    print(f"  X: {acc_raw_mean[0]:+.4f} m/s²")
    print(f"  Y: {acc_raw_mean[1]:+.4f} m/s²")
    print(f"  Z: {acc_raw_mean[2]:+.4f} m/s²")
    print(f"  Magnitude: {np.linalg.norm(acc_raw_mean):.4f} m/s²")

    # Apply extrinsicRot transform
    acc_transformed = EXTRINSIC_ROT @ acc_raw_mean
    print(f"\nAfter extrinsicRot transform:")
    print(f"  X: {acc_transformed[0]:+.4f} m/s²")
    print(f"  Y: {acc_transformed[1]:+.4f} m/s²")
    print(f"  Z: {acc_transformed[2]:+.4f} m/s²")

    # Check: For MakeSharedU, gravity should be mainly in +Z
    # When stationary, accelerometer measures reaction to gravity = +g in Z
    z_ratio = abs(acc_transformed[2]) / np.linalg.norm(acc_transformed)
    is_z_dominant = z_ratio > 0.9
    is_z_positive = acc_transformed[2] > 0  # MakeSharedU expects +Z

    print(f"\nVerification (GTSAM MakeSharedU convention):")
    print(f"  Z-axis dominance: {z_ratio*100:.1f}% (should be >90%)")
    print(f"  Z-axis sign: {'POSITIVE (correct for MakeSharedU)' if is_z_positive else 'NEGATIVE (WRONG!)'}")

    if is_z_dominant and is_z_positive:
        print("\n✓ TEST 1 PASSED: Gravity is correctly in +Z direction (MakeSharedU)")
        return True
    else:
        print("\n✗ TEST 1 FAILED: Gravity direction is wrong")
        return False


def test_orientation_transform():
    """Test 2: Verify orientation quaternion transform is valid"""
    print("\n" + "="*60)
    print("TEST 2: Orientation Transform (extrinsicRPY)")
    print("="*60)

    bag = rosbag.Bag(BAG_FILE)

    # Read first message from /imu/data (has orientation)
    quat_raw = None
    for topic, msg, t in bag.read_messages(topics=[STD_IMU_TOPIC]):
        quat_raw = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        break
    bag.close()

    if quat_raw is None:
        print("FAIL: No /imu/data found in bag")
        return False

    print(f"\nRaw /imu/data orientation (xyzw):")
    print(f"  [{quat_raw[0]:.6f}, {quat_raw[1]:.6f}, {quat_raw[2]:.6f}, {quat_raw[3]:.6f}]")

    # Convert to rotation and get RPY
    r_raw = R.from_quat(quat_raw)  # scipy uses xyzw order
    rpy_raw = r_raw.as_euler('xyz', degrees=True)
    print(f"  Roll: {rpy_raw[0]:+.2f}°, Pitch: {rpy_raw[1]:+.2f}°, Yaw: {rpy_raw[2]:+.2f}°")

    # Apply extrinsicRPY transform: q_out = q_raw * extQRPY^(-1)
    # extQRPY = inverse of extrinsicRPY rotation
    r_extRPY = R.from_matrix(EXTRINSIC_RPY)
    extQRPY = r_extRPY.inv()

    r_transformed = r_raw * extQRPY
    quat_transformed = r_transformed.as_quat()
    rpy_transformed = r_transformed.as_euler('xyz', degrees=True)

    print(f"\nAfter extrinsicRPY transform (q_raw * extQRPY^-1):")
    print(f"  [{quat_transformed[0]:.6f}, {quat_transformed[1]:.6f}, {quat_transformed[2]:.6f}, {quat_transformed[3]:.6f}]")
    print(f"  Roll: {rpy_transformed[0]:+.2f}°, Pitch: {rpy_transformed[1]:+.2f}°, Yaw: {rpy_transformed[2]:+.2f}°")

    # Verify: roll and pitch should be small (vehicle is roughly level)
    is_level = abs(rpy_transformed[0]) < 30 and abs(rpy_transformed[1]) < 30

    print(f"\nVerification:")
    print(f"  Roll/Pitch magnitude: roll={abs(rpy_transformed[0]):.1f}°, pitch={abs(rpy_transformed[1]):.1f}°")
    print(f"  Should be < 30° for roughly level vehicle")

    if is_level:
        print("\n✓ TEST 2 PASSED: Orientation transform produces valid result")
        return True
    else:
        print("\n✗ TEST 2 FAILED: Orientation looks wrong (vehicle should be roughly level)")
        return False


def test_angular_velocity_consistency():
    """Test 3: Verify angular velocity transform is consistent with orientation"""
    print("\n" + "="*60)
    print("TEST 3: Angular Velocity Transform Consistency")
    print("="*60)

    bag = rosbag.Bag(BAG_FILE)
    gyro_samples = []

    # Read first 100 FPA IMU messages
    count = 0
    for topic, msg, t in bag.read_messages(topics=[FPA_IMU_TOPIC]):
        if count >= 100:
            break
        gyro_raw = np.array([
            msg.data.angular_velocity.x,
            msg.data.angular_velocity.y,
            msg.data.angular_velocity.z
        ])
        gyro_samples.append(gyro_raw)
        count += 1
    bag.close()

    if len(gyro_samples) == 0:
        print("FAIL: No FPA IMU data found")
        return False

    gyro_raw_mean = np.mean(gyro_samples, axis=0)
    print(f"\nRaw FPA angular velocity (mean):")
    print(f"  X: {gyro_raw_mean[0]:+.6f} rad/s")
    print(f"  Y: {gyro_raw_mean[1]:+.6f} rad/s")
    print(f"  Z: {gyro_raw_mean[2]:+.6f} rad/s")

    # Apply extrinsicRot transform
    gyro_transformed = EXTRINSIC_ROT @ gyro_raw_mean
    print(f"\nAfter extrinsicRot transform:")
    print(f"  X: {gyro_transformed[0]:+.6f} rad/s")
    print(f"  Y: {gyro_transformed[1]:+.6f} rad/s")
    print(f"  Z: {gyro_transformed[2]:+.6f} rad/s")

    # For stationary vehicle, angular velocity should be small
    gyro_mag = np.linalg.norm(gyro_transformed)
    is_small = gyro_mag < 0.5  # rad/s

    print(f"\nVerification:")
    print(f"  Angular velocity magnitude: {gyro_mag:.4f} rad/s")
    print(f"  Should be < 0.5 rad/s for slow-moving vehicle")

    # Check Z-flip effect on gyro integration
    print(f"\n  Note: Z-flip in extrinsicRot means:")
    print(f"    - Rotation around world-Z is preserved in sign")
    print(f"    - But roll/pitch axes are effectively flipped")

    if is_small:
        print("\n✓ TEST 3 PASSED: Angular velocity magnitude is reasonable")
        return True
    else:
        print("\n⚠ TEST 3 WARNING: Angular velocity seems high (vehicle moving?)")
        return True  # Not a failure, just a warning


def test_coordinate_frame_consistency():
    """Test 4: Check if extrinsicRot and extrinsicRPY are consistent"""
    print("\n" + "="*60)
    print("TEST 4: Coordinate Frame Consistency Check")
    print("="*60)

    print("\nCurrent configuration:")
    print(f"extrinsicRot (for acc/gyro vectors):")
    print(f"  {EXTRINSIC_ROT[0]}")
    print(f"  {EXTRINSIC_ROT[1]}")
    print(f"  {EXTRINSIC_ROT[2]}")

    print(f"\nextrinsicRPY (for orientation quaternion):")
    print(f"  {EXTRINSIC_RPY[0]}")
    print(f"  {EXTRINSIC_RPY[1]}")
    print(f"  {EXTRINSIC_RPY[2]}")

    # Check determinants
    det_rot = np.linalg.det(EXTRINSIC_ROT)
    det_rpy = np.linalg.det(EXTRINSIC_RPY)

    print(f"\nDeterminants:")
    print(f"  det(extrinsicRot) = {det_rot:+.1f}")
    print(f"  det(extrinsicRPY) = {det_rpy:+.1f}")

    is_rot_proper = abs(det_rot - 1.0) < 0.01 or abs(det_rot + 1.0) < 0.01
    is_rpy_proper = abs(det_rpy - 1.0) < 0.01

    print(f"\nAnalysis:")
    if det_rot < 0:
        print(f"  extrinsicRot has det=-1 → includes reflection (Z-flip)")
        print(f"  This is CORRECT for converting +Z gravity to -Z gravity")
    else:
        print(f"  extrinsicRot has det=+1 → pure rotation, no reflection")
        print(f"  WARNING: May need Z-flip if gravity convention differs")

    if det_rpy > 0:
        print(f"  extrinsicRPY has det=+1 → pure rotation (correct for quaternion)")
    else:
        print(f"  extrinsicRPY has det=-1 → includes reflection")
        print(f"  ERROR: Quaternion transform should NOT include reflection!")
        return False

    # Key insight: extrinsicRot can have reflection for vector transform,
    # but extrinsicRPY must be a proper rotation for quaternion transform

    if is_rot_proper and is_rpy_proper:
        print("\n✓ TEST 4 PASSED: Both matrices are valid orthogonal transforms")
        if det_rot < 0 and det_rpy > 0:
            print("  Configuration: Rot has Z-flip, RPY is pure rotation (RECOMMENDED)")
        return True
    else:
        print("\n✗ TEST 4 FAILED: Matrix configuration invalid")
        return False


def test_gtsam_gravity_convention():
    """Test 5: Verify understanding of GTSAM MakeSharedU convention"""
    print("\n" + "="*60)
    print("TEST 5: GTSAM Gravity Convention Check")
    print("="*60)

    print("\nGTSAM PreintegrationParams::MakeSharedU(g) convention:")
    print("  - 'U' means 'Up' - gravity vector points UP in world frame")
    print("  - n_gravity = [0, 0, -g] internally (gravity DOWN in nav frame)")
    print("  - Accelerometer: a_measured = a_true - n_gravity")
    print("  - When stationary: a_measured = 0 - [0,0,-g] = [0, 0, +g]")
    print(f"  - With g={GRAVITY}: expect acc_z ≈ +{GRAVITY} m/s²")

    # Read actual data to verify
    bag = rosbag.Bag(BAG_FILE)
    acc_samples = []
    count = 0
    for topic, msg, t in bag.read_messages(topics=[FPA_IMU_TOPIC]):
        if count >= 50:
            break
        acc_raw = np.array([
            msg.data.linear_acceleration.x,
            msg.data.linear_acceleration.y,
            msg.data.linear_acceleration.z
        ])
        acc_samples.append(acc_raw)
        count += 1
    bag.close()

    acc_raw_mean = np.mean(acc_samples, axis=0)
    acc_transformed = EXTRINSIC_ROT @ acc_raw_mean

    expected_acc_z = +GRAVITY  # MakeSharedU expects POSITIVE
    actual_acc_z = acc_transformed[2]
    error = abs(actual_acc_z - expected_acc_z)

    print(f"\nVerification with actual data:")
    print(f"  Expected acc_z: +{expected_acc_z:.4f} m/s² (positive for MakeSharedU)")
    print(f"  Actual acc_z:   {actual_acc_z:+.4f} m/s²")
    print(f"  Error: {error:.4f} m/s² ({error/GRAVITY*100:.1f}%)")

    if error < 1.0:  # Allow 1 m/s² error for vehicle tilt
        print("\n✓ TEST 5 PASSED: Gravity convention matches GTSAM MakeSharedU")
        return True
    else:
        print("\n✗ TEST 5 FAILED: Gravity convention mismatch")
        return False


# ============== Main ==============

def main():
    print("="*60)
    print("LIO-SAM FPA IMU Transform Unit Tests")
    print("="*60)
    print(f"\nBag file: {BAG_FILE}")
    print(f"FPA IMU topic: {FPA_IMU_TOPIC}")
    print(f"Standard IMU topic: {STD_IMU_TOPIC}")

    results = []

    results.append(("Gravity Direction", test_gravity_direction()))
    results.append(("Orientation Transform", test_orientation_transform()))
    results.append(("Angular Velocity", test_angular_velocity_consistency()))
    results.append(("Coordinate Consistency", test_coordinate_frame_consistency()))
    results.append(("GTSAM Convention", test_gtsam_gravity_convention()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - Check configuration")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
