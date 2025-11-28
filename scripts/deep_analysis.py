#!/usr/bin/env python3
"""
Deep analysis of FPA odometry orientation and frame consistency
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R

def analyze_fpa_orientation_deeply(bag_path, topic='/fixposition/fpa/odometry', skip_samples=0):
    """Deep analysis of FPA orientation"""
    print("\n" + "="*60)
    print("FPA Odometry Deep Analysis")
    print("="*60)

    bag = rosbag.Bag(bag_path, 'r')
    positions = []
    orientations = []
    timestamps = []
    count = 0

    # Read more samples to see movement
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if count < skip_samples:
            count += 1
            continue

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

        if count >= skip_samples + 5000:  # Analyze 5000 samples
            break

    bag.close()

    positions = np.array(positions)
    orientations = np.array(orientations)
    timestamps = np.array(timestamps)

    print(f"\nSamples analyzed: {len(positions)}")
    print(f"Time range: {timestamps[0]:.2f} to {timestamps[-1]:.2f} ({timestamps[-1]-timestamps[0]:.2f}s)")

    # ECEF position analysis
    print("\n--- ECEF Position Analysis ---")
    print(f"X: min={positions[:,0].min():.2f}, max={positions[:,0].max():.2f}, range={positions[:,0].max()-positions[:,0].min():.2f}m")
    print(f"Y: min={positions[:,1].min():.2f}, max={positions[:,1].max():.2f}, range={positions[:,1].max()-positions[:,1].min():.2f}m")
    print(f"Z: min={positions[:,2].min():.2f}, max={positions[:,2].max():.2f}, range={positions[:,2].max()-positions[:,2].min():.2f}m")

    # First position
    origin_ecef = positions[0]
    lat, lon, alt = ecef_to_lla(origin_ecef[0], origin_ecef[1], origin_ecef[2])
    print(f"\nOrigin (LLA): lat={lat:.6f}°, lon={lon:.6f}°, alt={alt:.2f}m")

    # Convert to ENU
    R_ecef_to_enu = get_ecef_to_enu_matrix(lat, lon)
    enu_positions = []
    for pos in positions:
        delta = pos - origin_ecef
        enu = R_ecef_to_enu @ delta
        enu_positions.append(enu)
    enu_positions = np.array(enu_positions)

    print("\n--- ENU Position Analysis ---")
    print(f"E: min={enu_positions[:,0].min():.2f}, max={enu_positions[:,0].max():.2f}m")
    print(f"N: min={enu_positions[:,1].min():.2f}, max={enu_positions[:,1].max():.2f}m")
    print(f"U: min={enu_positions[:,2].min():.2f}, max={enu_positions[:,2].max():.2f}m")

    # Analyze orientation
    print("\n--- Orientation Analysis ---")

    # Convert quaternions to RPY
    rpys = []
    for q in orientations:
        r = R.from_quat(q)  # scipy uses [x,y,z,w]
        rpy = r.as_euler('xyz', degrees=True)
        rpys.append(rpy)
    rpys = np.array(rpys)

    print(f"Roll: min={rpys[:,0].min():.2f}°, max={rpys[:,0].max():.2f}°, mean={rpys[:,0].mean():.2f}°")
    print(f"Pitch: min={rpys[:,1].min():.2f}°, max={rpys[:,1].max():.2f}°, mean={rpys[:,1].mean():.2f}°")
    print(f"Yaw: min={rpys[:,2].min():.2f}°, max={rpys[:,2].max():.2f}°, mean={rpys[:,2].mean():.2f}°")

    # Check if orientation changes match position changes
    print("\n--- Movement vs Orientation Correlation ---")
    if len(enu_positions) > 100:
        # Calculate velocity direction from position changes
        dt = np.diff(timestamps)
        dpos = np.diff(enu_positions, axis=0)
        vel_e = dpos[:,0] / dt
        vel_n = dpos[:,1] / dt

        # Filter out stationary periods
        speed = np.sqrt(vel_e**2 + vel_n**2)
        moving_mask = speed > 0.5  # > 0.5 m/s

        if np.sum(moving_mask) > 10:
            # Calculate heading from velocity (direction of movement)
            heading_from_vel = np.arctan2(vel_e[moving_mask], vel_n[moving_mask]) * 180 / np.pi

            # Get yaw from orientation at same times
            yaw_from_orient = rpys[1:][moving_mask, 2]  # Yaw from orientation

            print(f"Moving samples: {np.sum(moving_mask)}")
            print(f"Speed range: {speed[moving_mask].min():.2f} to {speed[moving_mask].max():.2f} m/s")
            print(f"Heading from velocity: mean={heading_from_vel.mean():.2f}°, std={heading_from_vel.std():.2f}°")
            print(f"Yaw from orientation: mean={yaw_from_orient.mean():.2f}°, std={yaw_from_orient.std():.2f}°")

    # Analyze the quaternion w.r.t ECEF frame
    print("\n--- ECEF Frame Analysis ---")
    print("The FPA orientation quaternion might be expressing the body frame w.r.t. ECEF frame")
    print("Let's check if converting to local ENU makes sense...")

    # For first sample, convert orientation from ECEF to ENU frame
    q_ecef = R.from_quat(orientations[0])

    # The rotation from ECEF to ENU at this location
    R_ecef_enu = R.from_matrix(R_ecef_to_enu)

    # If q_ecef represents body w.r.t. ECEF, then body w.r.t. ENU would be:
    # q_enu = R_ecef_enu * q_ecef
    q_enu = R_ecef_enu * q_ecef

    rpy_enu = q_enu.as_euler('xyz', degrees=True)
    print(f"Original FPA RPY (ECEF?): Roll={rpys[0,0]:.2f}°, Pitch={rpys[0,1]:.2f}°, Yaw={rpys[0,2]:.2f}°")
    print(f"Converted to ENU frame:   Roll={rpy_enu[0]:.2f}°, Pitch={rpy_enu[1]:.2f}°, Yaw={rpy_enu[2]:.2f}°")

    return positions, enu_positions, orientations


def analyze_imu_vs_fpa(bag_path, num_samples=1000):
    """Compare IMU orientation with FPA orientation"""
    print("\n" + "="*60)
    print("IMU vs FPA Orientation Comparison")
    print("="*60)

    bag = rosbag.Bag(bag_path, 'r')

    # Collect IMU data
    imu_data = []
    for topic_name, msg, t in bag.read_messages(topics=['/imu/data']):
        if len(imu_data) >= num_samples:
            break
        imu_data.append({
            'time': t.to_sec(),
            'quat': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'accel': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        })

    # Collect FPA data
    fpa_data = []
    for topic_name, msg, t in bag.read_messages(topics=['/fixposition/fpa/odometry']):
        if len(fpa_data) >= num_samples:
            break
        fpa_data.append({
            'time': t.to_sec(),
            'quat': [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],
            'pos': [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        })

    bag.close()

    print(f"\nIMU samples: {len(imu_data)}")
    print(f"FPA samples: {len(fpa_data)}")

    if imu_data and fpa_data:
        # Compare first few orientations
        print("\n--- First 5 samples comparison ---")
        for i in range(min(5, len(imu_data), len(fpa_data))):
            imu_q = imu_data[i]['quat']
            fpa_q = fpa_data[i]['quat']

            imu_rpy = R.from_quat(imu_q).as_euler('xyz', degrees=True)
            fpa_rpy = R.from_quat(fpa_q).as_euler('xyz', degrees=True)

            print(f"Sample {i}:")
            print(f"  IMU: Roll={imu_rpy[0]:.2f}°, Pitch={imu_rpy[1]:.2f}°, Yaw={imu_rpy[2]:.2f}°")
            print(f"  FPA: Roll={fpa_rpy[0]:.2f}°, Pitch={fpa_rpy[1]:.2f}°, Yaw={fpa_rpy[2]:.2f}°")
            print(f"  Diff: ΔRoll={imu_rpy[0]-fpa_rpy[0]:.2f}°, ΔPitch={imu_rpy[1]-fpa_rpy[1]:.2f}°, ΔYaw={imu_rpy[2]-fpa_rpy[2]:.2f}°")


def analyze_frame_ids(bag_path):
    """Check frame_id consistency across topics"""
    print("\n" + "="*60)
    print("Frame ID Analysis")
    print("="*60)

    bag = rosbag.Bag(bag_path, 'r')

    frame_ids = {
        '/imu/data': set(),
        '/lidar_points': set(),
        '/fixposition/fpa/odometry': set()
    }

    for topic in frame_ids.keys():
        count = 0
        for topic_name, msg, t in bag.read_messages(topics=[topic]):
            if hasattr(msg, 'header'):
                frame_ids[topic].add(msg.header.frame_id)
            elif hasattr(msg, 'pose'):
                if hasattr(msg.pose, 'header'):
                    frame_ids[topic].add(msg.pose.header.frame_id)
            count += 1
            if count > 10:
                break

    bag.close()

    print("\nFrame IDs found:")
    for topic, frames in frame_ids.items():
        print(f"  {topic}: {frames if frames else 'N/A'}")

    print("\n--- Frame ID Configuration Issues ---")
    print("params.yaml configuration:")
    print("  lidarFrame: base_link")
    print("  baselinkFrame: base_link")
    print("")
    print("Actual data frame IDs:")
    print(f"  LiDAR: {frame_ids['/lidar_points']}")
    print("")
    if 'lidar_link' in frame_ids['/lidar_points']:
        print("⚠ WARNING: LiDAR frame is 'lidar_link' but config expects 'base_link'")
        print("  This mismatch may cause TF lookup issues!")
        print("  Consider changing lidarFrame in params.yaml to 'lidar_link'")
        print("  OR publish a static TF from lidar_link to base_link")


def ecef_to_lla(x, y, z):
    """Convert ECEF to latitude, longitude, altitude (WGS84)"""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2.0 * f - f * f

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - e2))

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

    R_mat = np.array([
        [-sin_lon,           cos_lon,          0],
        [-sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat],
        [ cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])

    return R_mat


def main():
    bag_path = "/root/autodl-tmp/info_fixed.bag"

    # Frame ID analysis
    analyze_frame_ids(bag_path)

    # IMU vs FPA comparison
    analyze_imu_vs_fpa(bag_path)

    # Deep FPA analysis (skip first 1000 samples to see movement)
    analyze_fpa_orientation_deeply(bag_path, skip_samples=5000)

    print("\n" + "="*60)
    print("Key Findings and Issues")
    print("="*60)

    print("""
1. FRAME ID MISMATCH:
   - LiDAR publishes to 'lidar_link' frame
   - params.yaml expects 'base_link' frame
   - Fix: Either change lidarFrame in params.yaml OR add static TF

2. FPA ORIENTATION FRAME:
   - FPA orientation appears to be in ECEF reference frame
   - fpaOdomConverter.cpp keeps orientation unchanged
   - This may cause issues since LIO-SAM expects local ENU orientation

3. IMU VS FPA ORIENTATION MISMATCH:
   - IMU shows yaw ≈ 84° (local frame, roughly east)
   - FPA shows yaw ≈ -154° (likely ECEF frame)
   - The FPA orientation needs proper conversion to local ENU

4. RECOMMENDED FIXES:
   a) Update fpaOdomConverter.cpp to properly convert FPA orientation from ECEF to ENU
   b) Update params.yaml lidarFrame to 'lidar_link' or add TF
   c) Verify IMU and LiDAR extrinsics if they are not co-located
""")


if __name__ == "__main__":
    main()
