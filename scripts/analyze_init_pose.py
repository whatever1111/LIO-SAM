#!/usr/bin/env python3
"""
Analyze initial pose from different sources at system startup.
Compare: GPS, /imu/data, FPA CORRIMU
Use GPS position/heading as ground truth reference.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import rosbag
import sys

BAG_FILE = "/root/autodl-tmp/info_fixed.bag"

# Topics
GPS_ODOM_TOPIC = "/fixposition/fpa/odometry"  # FPA odometry with position and orientation
IMU_DATA_TOPIC = "/imu/data"  # Standard IMU with orientation
FPA_IMU_TOPIC = "/fixposition/fpa/corrimu"  # CORRIMU (no orientation)

# Extrinsic transform (Rz(90))
EXTRINSIC_ROT = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])

def quat_to_rpy_deg(qx, qy, qz, qw):
    """Convert quaternion to roll, pitch, yaw in degrees"""
    r = R.from_quat([qx, qy, qz, qw])
    rpy = r.as_euler('xyz', degrees=True)
    return rpy[0], rpy[1], rpy[2]

def main():
    print("="*70)
    print("Initial Pose Analysis - GPS as Ground Truth")
    print("="*70)

    bag = rosbag.Bag(BAG_FILE)

    # Get first few messages from each topic
    gps_msgs = []
    imu_msgs = []
    corrimu_msgs = []

    # Read first 10 messages from each topic
    for topic, msg, t in bag.read_messages(topics=[GPS_ODOM_TOPIC]):
        gps_msgs.append((t.to_sec(), msg))
        if len(gps_msgs) >= 10:
            break

    for topic, msg, t in bag.read_messages(topics=[IMU_DATA_TOPIC]):
        imu_msgs.append((t.to_sec(), msg))
        if len(imu_msgs) >= 10:
            break

    for topic, msg, t in bag.read_messages(topics=[FPA_IMU_TOPIC]):
        corrimu_msgs.append((t.to_sec(), msg))
        if len(corrimu_msgs) >= 10:
            break

    bag.close()

    # ============ GPS Analysis ============
    print("\n" + "="*70)
    print("1. GPS/FPA Odometry (Ground Truth)")
    print("="*70)

    if gps_msgs:
        t, msg = gps_msgs[0]
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        print(f"\nFirst GPS message at t={t:.3f}")
        print(f"Position (ECEF): X={pos.x:.3f}, Y={pos.y:.3f}, Z={pos.z:.3f}")

        roll, pitch, yaw = quat_to_rpy_deg(ori.x, ori.y, ori.z, ori.w)
        print(f"Orientation (ENU): Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f} deg")
        print(f"Quaternion (xyzw): [{ori.x:.6f}, {ori.y:.6f}, {ori.z:.6f}, {ori.w:.6f}]")

        # GPS heading is typically yaw in ENU frame
        gps_heading = yaw
        print(f"\n>>> GPS Heading (Yaw in ENU): {gps_heading:.2f} degrees")

    else:
        print("No GPS messages found!")
        gps_heading = None

    # ============ /imu/data Analysis ============
    print("\n" + "="*70)
    print("2. /imu/data Orientation (Fixposition INS Fusion)")
    print("="*70)

    if imu_msgs:
        t, msg = imu_msgs[0]
        ori = msg.orientation

        print(f"\nFirst /imu/data message at t={t:.3f}")

        # Raw orientation
        roll_raw, pitch_raw, yaw_raw = quat_to_rpy_deg(ori.x, ori.y, ori.z, ori.w)
        print(f"Raw Orientation: Roll={roll_raw:.2f}, Pitch={pitch_raw:.2f}, Yaw={yaw_raw:.2f} deg")
        print(f"Raw Quaternion (xyzw): [{ori.x:.6f}, {ori.y:.6f}, {ori.z:.6f}, {ori.w:.6f}]")

        # After extrinsicRPY transform: q_out = q_raw * extQRPY^(-1)
        r_raw = R.from_quat([ori.x, ori.y, ori.z, ori.w])
        r_extRPY = R.from_matrix(EXTRINSIC_ROT)
        extQRPY = r_extRPY.inv()

        r_transformed = r_raw * extQRPY
        rpy_transformed = r_transformed.as_euler('xyz', degrees=True)

        print(f"\nAfter extrinsicRPY transform (Rz(90)^-1 = Rz(-90)):")
        print(f"Transformed: Roll={rpy_transformed[0]:.2f}, Pitch={rpy_transformed[1]:.2f}, Yaw={rpy_transformed[2]:.2f} deg")

        imu_heading_raw = yaw_raw
        imu_heading_transformed = rpy_transformed[2]
        print(f"\n>>> /imu/data Raw Yaw: {imu_heading_raw:.2f} degrees")
        print(f">>> /imu/data Transformed Yaw: {imu_heading_transformed:.2f} degrees")

    else:
        print("No /imu/data messages found!")
        imu_heading_raw = None
        imu_heading_transformed = None

    # ============ CORRIMU Analysis ============
    print("\n" + "="*70)
    print("3. CORRIMU Data (No Orientation)")
    print("="*70)

    if corrimu_msgs:
        # Average first few messages
        acc_samples = []
        gyro_samples = []

        for t, msg in corrimu_msgs[:5]:
            acc_samples.append([
                msg.data.linear_acceleration.x,
                msg.data.linear_acceleration.y,
                msg.data.linear_acceleration.z
            ])
            gyro_samples.append([
                msg.data.angular_velocity.x,
                msg.data.angular_velocity.y,
                msg.data.angular_velocity.z
            ])

        acc_mean = np.mean(acc_samples, axis=0)
        gyro_mean = np.mean(gyro_samples, axis=0)

        print(f"\nCORRIMU (average of first 5 messages):")
        print(f"Raw Acceleration: X={acc_mean[0]:.4f}, Y={acc_mean[1]:.4f}, Z={acc_mean[2]:.4f} m/s²")
        print(f"Raw Angular Vel:  X={gyro_mean[0]:.6f}, Y={gyro_mean[1]:.6f}, Z={gyro_mean[2]:.6f} rad/s")

        # After extrinsicRot transform
        acc_transformed = EXTRINSIC_ROT @ acc_mean
        gyro_transformed = EXTRINSIC_ROT @ gyro_mean

        print(f"\nAfter extrinsicRot transform (Rz(90)):")
        print(f"Transformed Acc:  X={acc_transformed[0]:.4f}, Y={acc_transformed[1]:.4f}, Z={acc_transformed[2]:.4f} m/s²")
        print(f"Transformed Gyro: X={gyro_transformed[0]:.6f}, Y={gyro_transformed[1]:.6f}, Z={gyro_transformed[2]:.6f} rad/s")

        # Check gravity direction
        gravity_mag = np.linalg.norm(acc_transformed)
        print(f"\nGravity magnitude: {gravity_mag:.4f} m/s² (should be ~9.8)")
        print(f"Gravity Z component: {acc_transformed[2]:.4f} m/s² (MakeSharedU expects +9.8)")

    else:
        print("No CORRIMU messages found!")

    # ============ Comparison ============
    print("\n" + "="*70)
    print("4. Heading Comparison")
    print("="*70)

    if gps_heading is not None and imu_heading_raw is not None:
        diff_raw = imu_heading_raw - gps_heading
        # Normalize to [-180, 180]
        while diff_raw > 180: diff_raw -= 360
        while diff_raw < -180: diff_raw += 360

        diff_transformed = imu_heading_transformed - gps_heading
        while diff_transformed > 180: diff_transformed -= 360
        while diff_transformed < -180: diff_transformed += 360

        print(f"\nGPS Heading (ground truth):     {gps_heading:.2f} deg")
        print(f"/imu/data Raw Yaw:              {imu_heading_raw:.2f} deg (diff: {diff_raw:+.2f} deg)")
        print(f"/imu/data Transformed Yaw:      {imu_heading_transformed:.2f} deg (diff: {diff_transformed:+.2f} deg)")

        print(f"\n>>> Heading Difference Analysis:")
        print(f"    Raw /imu/data vs GPS:         {diff_raw:+.2f} degrees")
        print(f"    Transformed /imu/data vs GPS: {diff_transformed:+.2f} degrees")

        # Check what extrinsic correction is needed
        if abs(diff_raw) < 10:
            print(f"\n    ANALYSIS: /imu/data raw heading is close to GPS!")
            print(f"              extrinsicRPY may NOT be needed for heading")
        elif abs(diff_transformed) < 10:
            print(f"\n    ANALYSIS: /imu/data transformed heading matches GPS!")
            print(f"              extrinsicRPY = Rz(90) is CORRECT")
        else:
            needed_correction = -diff_raw
            print(f"\n    ANALYSIS: Neither raw nor transformed matches GPS well")
            print(f"              Suggested heading correction: {needed_correction:+.2f} degrees")
            print(f"              This corresponds to Rz({needed_correction:.0f})")

    # ============ LIO-SAM Initialization Analysis ============
    print("\n" + "="*70)
    print("5. LIO-SAM Initialization Path Analysis")
    print("="*70)

    print("""
LIO-SAM Initial Heading Flow:

1. imageProjection.cpp:imuDeskewInfo()
   - Gets imuRollInit, imuPitchInit, imuYawInit from first IMU message
   - Uses imuRPY2rosRPY() which applies extrinsicRPY transform

2. mapOptmization.cpp:updateInitialGuess()
   - Sets transformTobeMapped[0,1,2] = cloudInfo.imuRollInit/Pitch/Yaw
   - This becomes the initial LiDAR pose

3. imuPreintegration.cpp:odometryHandler()
   - On first LiDAR odometry, initializes corrImuOrientation_ from LiDAR pose
   - This sets the IMU integration starting point

Key Question: Is the extrinsicRPY transform producing the correct initial yaw?
""")

    if gps_heading is not None and imu_heading_transformed is not None:
        expected_lio_init_yaw = imu_heading_transformed
        print(f"Expected LIO-SAM initial yaw (from /imu/data after transform): {expected_lio_init_yaw:.2f} deg")
        print(f"GPS ground truth heading: {gps_heading:.2f} deg")
        print(f"Difference: {diff_transformed:+.2f} deg")

        if abs(diff_transformed) > 30:
            print(f"\n!!! WARNING: Large heading difference ({abs(diff_transformed):.1f}°) !!!")
            print(f"    This will cause trajectory to diverge from GPS!")

    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)

if __name__ == "__main__":
    main()
