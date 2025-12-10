#!/usr/bin/env python3
"""
Analyze GPS/FPA Odometry frame and orientation convention.
The /fixposition/fpa/odometry may use ECEF coordinates, not ENU.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import rosbag

BAG_FILE = "/root/autodl-tmp/info_fixed.bag"
GPS_ODOM_TOPIC = "/fixposition/fpa/odometry"
IMU_DATA_TOPIC = "/imu/data"

def ecef_to_lla(x, y, z):
    """Convert ECEF to Latitude, Longitude, Altitude (WGS84)"""
    # WGS84 constants
    a = 6378137.0  # semi-major axis
    f = 1/298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    e2 = (a**2 - b**2) / a**2  # first eccentricity squared
    ep2 = (a**2 - b**2) / b**2  # second eccentricity squared

    p = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)

    # Iterative calculation for latitude
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), p)

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt

def ecef_to_enu_rotation(lat, lon):
    """Get rotation matrix from ECEF to ENU at given lat/lon"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # ECEF to ENU rotation matrix
    R_ecef_to_enu = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad)*np.cos(lon_rad), np.cos(lat_rad)*np.sin(lon_rad), np.sin(lat_rad)]
    ])
    return R_ecef_to_enu

def main():
    print("="*70)
    print("GPS Frame Analysis - ECEF vs ENU")
    print("="*70)

    bag = rosbag.Bag(BAG_FILE)

    # Get first GPS message
    gps_msg = None
    for topic, msg, t in bag.read_messages(topics=[GPS_ODOM_TOPIC]):
        gps_msg = msg
        gps_time = t.to_sec()
        break

    # Get first /imu/data message
    imu_msg = None
    for topic, msg, t in bag.read_messages(topics=[IMU_DATA_TOPIC]):
        imu_msg = msg
        imu_time = t.to_sec()
        break

    bag.close()

    if gps_msg is None:
        print("No GPS data found!")
        return

    # ============ Analyze GPS Position ============
    print("\n" + "="*70)
    print("1. GPS Position Analysis")
    print("="*70)

    pos = gps_msg.pose.pose.position
    print(f"\nRaw position: X={pos.x:.3f}, Y={pos.y:.3f}, Z={pos.z:.3f}")

    # Check if this looks like ECEF (magnitude ~6.4e6 meters from Earth center)
    r = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
    print(f"Distance from origin: {r:.3f} m = {r/1e6:.3f} x 10^6 m")

    if r > 6e6:
        print(">>> Position is in ECEF coordinates (Earth-Centered Earth-Fixed)")

        lat, lon, alt = ecef_to_lla(pos.x, pos.y, pos.z)
        print(f"\nConverted to LLA:")
        print(f"  Latitude:  {lat:.6f}°")
        print(f"  Longitude: {lon:.6f}°")
        print(f"  Altitude:  {alt:.2f} m")

        # Get ECEF to ENU rotation at this location
        R_ecef_enu = ecef_to_enu_rotation(lat, lon)
    else:
        print(">>> Position appears to be in local frame (not ECEF)")
        R_ecef_enu = np.eye(3)
        lat, lon = 0, 0

    # ============ Analyze GPS Orientation ============
    print("\n" + "="*70)
    print("2. GPS Orientation Analysis")
    print("="*70)

    ori = gps_msg.pose.pose.orientation
    print(f"\nRaw quaternion (xyzw): [{ori.x:.6f}, {ori.y:.6f}, {ori.z:.6f}, {ori.w:.6f}]")

    # Convert to rotation matrix
    r_raw = R.from_quat([ori.x, ori.y, ori.z, ori.w])
    rpy_raw = r_raw.as_euler('xyz', degrees=True)
    print(f"As Euler (xyz): Roll={rpy_raw[0]:.2f}, Pitch={rpy_raw[1]:.2f}, Yaw={rpy_raw[2]:.2f} deg")

    # The FPA odometry orientation is in ECEF frame
    # To get heading in ENU, we need to transform
    if r > 6e6:
        print("\n>>> Orientation is likely in ECEF frame")
        print(">>> Converting to ENU frame...")

        R_body_ecef = r_raw.as_matrix()

        # R_body_enu = R_ecef_enu @ R_body_ecef
        R_body_enu = R_ecef_enu @ R_body_ecef

        r_enu = R.from_matrix(R_body_enu)
        rpy_enu = r_enu.as_euler('xyz', degrees=True)
        print(f"ENU Euler (xyz): Roll={rpy_enu[0]:.2f}, Pitch={rpy_enu[1]:.2f}, Yaw={rpy_enu[2]:.2f} deg")

        # Yaw in ENU is heading (0=East, 90=North, -90=South)
        heading_enu = rpy_enu[2]
        # Convert to heading from North (0=North, 90=East)
        heading_from_north = 90 - heading_enu
        print(f"\n>>> Heading in ENU (from East): {heading_enu:.2f} deg")
        print(f">>> Heading from North: {heading_from_north:.2f} deg")

    # ============ Compare with /imu/data ============
    print("\n" + "="*70)
    print("3. /imu/data Orientation (should be in ENU already)")
    print("="*70)

    if imu_msg:
        ori_imu = imu_msg.orientation
        print(f"\nQuaternion (xyzw): [{ori_imu.x:.6f}, {ori_imu.y:.6f}, {ori_imu.z:.6f}, {ori_imu.w:.6f}]")

        r_imu = R.from_quat([ori_imu.x, ori_imu.y, ori_imu.z, ori_imu.w])
        rpy_imu = r_imu.as_euler('xyz', degrees=True)
        print(f"Euler (xyz): Roll={rpy_imu[0]:.2f}, Pitch={rpy_imu[1]:.2f}, Yaw={rpy_imu[2]:.2f} deg")

        imu_heading = rpy_imu[2]
        print(f"\n>>> /imu/data Yaw: {imu_heading:.2f} deg")

        # Check if /imu/data frame_id gives hints
        print(f"\n/imu/data frame_id: {imu_msg.header.frame_id}")

    # ============ Frame Analysis ============
    print("\n" + "="*70)
    print("4. Frame Analysis Summary")
    print("="*70)

    print("""
Fixposition FPA data frames:
- /fixposition/fpa/odometry: Position and orientation in ECEF frame
- /imu/data: IMU data with orientation (frame depends on driver config)

Key insight: The orientation in ECEF frame cannot be directly compared
to heading in local ENU frame without coordinate transformation!

For LIO-SAM:
- Initial heading should come from /imu/data (already in body/ENU frame)
- GPS factor uses position only, not orientation (in most configurations)
""")

    if r > 6e6 and imu_msg:
        print(f"\nHeading comparison:")
        print(f"  GPS (converted to ENU): {heading_enu:.2f} deg")
        print(f"  /imu/data yaw:          {imu_heading:.2f} deg")
        diff = heading_enu - imu_heading
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        print(f"  Difference: {diff:.2f} deg")

        if abs(diff) < 20:
            print("\n>>> GPS ENU heading and /imu/data yaw are reasonably close!")
        else:
            print(f"\n>>> Large difference ({abs(diff):.1f}°) - check frame conventions")

if __name__ == "__main__":
    main()
