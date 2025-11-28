#!/usr/bin/env python3
"""
LIO-SAM Full System Validator
=============================
Validates all components of the LIO-SAM system:
1. IMU data quality and coordinate frame
2. LiDAR data format and timing
3. GPS/ECEF to ENU conversion
4. Coordinate frame alignment
5. Velocity estimation
6. Time synchronization

Usage:
    python3 full_system_validator.py /path/to/bag_file.bag
"""

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
from collections import defaultdict
import struct

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_ok(text):
    print(f"{Colors.GREEN}[OK]{Colors.ENDC} {text}")

def print_warn(text):
    print(f"{Colors.WARNING}[WARN]{Colors.ENDC} {text}")

def print_fail(text):
    print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {text}")

def print_info(text):
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")


class FullSystemValidator:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.results = {}

    def run_all_validations(self):
        print_header("LIO-SAM Full System Validator")
        print(f"Bag file: {self.bag_path}\n")

        # 1. Basic bag info
        self.validate_bag_info()

        # 2. IMU validation
        self.validate_imu()

        # 3. LiDAR validation
        self.validate_lidar()

        # 4. GPS/FPA validation
        self.validate_gps()

        # 5. Time synchronization
        self.validate_time_sync()

        # 6. Coordinate frame validation
        self.validate_coordinate_frames()

        # 7. Velocity estimation test
        self.validate_velocity_estimation()

        # Summary
        self.print_summary()

    def validate_bag_info(self):
        print_header("1. Bag File Information")

        bag = rosbag.Bag(self.bag_path)
        info = bag.get_type_and_topic_info()

        print(f"Duration: {bag.get_end_time() - bag.get_start_time():.2f} seconds")
        print(f"Start time: {bag.get_start_time()}")
        print(f"End time: {bag.get_end_time()}")
        print(f"\nTopics:")

        required_topics = {
            '/imu/data': False,
            '/lidar_points': False,
            '/fixposition/fpa/odometry': False
        }

        for topic, info in info.topics.items():
            status = "required" if topic in required_topics else ""
            if topic in required_topics:
                required_topics[topic] = True
            print(f"  {topic}: {info.message_count} msgs, {info.msg_type} {status}")

        bag.close()

        # Check required topics
        all_found = True
        for topic, found in required_topics.items():
            if found:
                print_ok(f"Required topic found: {topic}")
            else:
                print_fail(f"Required topic missing: {topic}")
                all_found = False

        self.results['bag_info'] = all_found

    def validate_imu(self):
        print_header("2. IMU Data Validation")

        bag = rosbag.Bag(self.bag_path)

        imu_data = []
        timestamps = []

        for topic, msg, t in bag.read_messages(topics=['/imu/data']):
            imu_data.append({
                'time': msg.header.stamp.to_sec(),
                'acc': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                'gyro': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'quat': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            })
            timestamps.append(msg.header.stamp.to_sec())
            if len(imu_data) > 5000:
                break

        bag.close()

        if len(imu_data) < 100:
            print_fail(f"Insufficient IMU data: {len(imu_data)} samples")
            self.results['imu'] = False
            return

        # Calculate statistics
        acc = np.array([d['acc'] for d in imu_data])
        gyro = np.array([d['gyro'] for d in imu_data])

        # IMU frequency
        dt = np.diff(timestamps)
        freq = 1.0 / np.mean(dt)
        print_info(f"IMU frequency: {freq:.1f} Hz")

        if freq < 100:
            print_warn(f"IMU frequency low (expected ~200Hz)")
        else:
            print_ok(f"IMU frequency acceptable")

        # Check for timestamp jumps
        max_dt = np.max(dt)
        if max_dt > 0.1:
            print_warn(f"Large timestamp gap detected: {max_dt:.3f}s")
        else:
            print_ok(f"Timestamps continuous (max gap: {max_dt*1000:.1f}ms)")

        # Accelerometer statistics
        acc_mean = np.mean(acc, axis=0)
        acc_std = np.std(acc, axis=0)
        acc_norm = np.linalg.norm(acc_mean)

        print(f"\nAccelerometer statistics:")
        print(f"  Mean: [{acc_mean[0]:.4f}, {acc_mean[1]:.4f}, {acc_mean[2]:.4f}] m/s^2")
        print(f"  Std:  [{acc_std[0]:.4f}, {acc_std[1]:.4f}, {acc_std[2]:.4f}] m/s^2")
        print(f"  Norm: {acc_norm:.4f} m/s^2 (expected ~9.81)")

        # Check gravity direction
        gravity_axis = np.argmax(np.abs(acc_mean))
        axis_names = ['X', 'Y', 'Z']
        print(f"  Gravity mainly in {axis_names[gravity_axis]} axis")

        if abs(acc_norm - 9.81) > 0.5:
            print_warn(f"Accelerometer norm deviation from gravity: {abs(acc_norm - 9.81):.2f} m/s^2")
        else:
            print_ok(f"Accelerometer norm close to gravity")

        # Check IMU coordinate frame (gravity should be in Z for NED/ENU)
        if gravity_axis == 2:
            if acc_mean[2] > 0:
                print_ok("IMU frame: Z-up (ENU-like), gravity = +Z")
            else:
                print_ok("IMU frame: Z-down (NED-like), gravity = -Z")
        else:
            print_warn(f"IMU frame unusual: gravity mainly in {axis_names[gravity_axis]} axis")

        # Gyroscope statistics (should be near zero when stationary)
        gyro_mean = np.mean(gyro, axis=0)
        gyro_std = np.std(gyro, axis=0)

        print(f"\nGyroscope statistics:")
        print(f"  Mean: [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}] rad/s")
        print(f"  Std:  [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}] rad/s")

        # Check orientation quaternion
        quats = np.array([d['quat'] for d in imu_data[:100]])
        quat_norms = np.linalg.norm(quats, axis=1)

        if np.allclose(quat_norms, 1.0, atol=0.01):
            print_ok("Quaternions normalized correctly")
        else:
            print_warn(f"Quaternion norm deviation: mean={np.mean(quat_norms):.4f}")

        # Extract initial yaw from quaternion
        r = R.from_quat(quats[0])
        euler = r.as_euler('xyz', degrees=True)
        print(f"\nInitial IMU orientation (from quaternion):")
        print(f"  Roll:  {euler[0]:.2f} deg")
        print(f"  Pitch: {euler[1]:.2f} deg")
        print(f"  Yaw:   {euler[2]:.2f} deg")

        self.results['imu'] = True
        self.imu_initial_yaw = euler[2]

    def validate_lidar(self):
        print_header("3. LiDAR Data Validation")

        bag = rosbag.Bag(self.bag_path)

        lidar_msgs = []
        for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
            lidar_msgs.append({
                'time': msg.header.stamp.to_sec(),
                'frame_id': msg.header.frame_id,
                'width': msg.width,
                'height': msg.height,
                'point_step': msg.point_step,
                'row_step': msg.row_step,
                'fields': [(f.name, f.offset, f.datatype) for f in msg.fields],
                'is_dense': msg.is_dense
            })
            if len(lidar_msgs) > 50:
                break

        bag.close()

        if len(lidar_msgs) == 0:
            print_fail("No LiDAR messages found")
            self.results['lidar'] = False
            return

        # Analyze first message
        msg = lidar_msgs[0]
        print(f"Frame ID: {msg['frame_id']}")
        print(f"Points per scan: {msg['width'] * msg['height']}")
        print(f"Point step: {msg['point_step']} bytes")
        print(f"Is dense: {msg['is_dense']}")

        print(f"\nPoint cloud fields:")
        for name, offset, dtype in msg['fields']:
            dtype_names = {1: 'INT8', 2: 'UINT8', 3: 'INT16', 4: 'UINT16',
                          5: 'INT32', 6: 'UINT32', 7: 'FLOAT32', 8: 'FLOAT64'}
            print(f"  {name}: offset={offset}, type={dtype_names.get(dtype, dtype)}")

        # Check for required fields
        field_names = [f[0] for f in msg['fields']]

        # Detect sensor type
        if 'ring' in field_names and 'time' in field_names:
            print_ok("Detected Velodyne-style format (ring + time)")
            sensor_type = "velodyne"
        elif 'line' in field_names and 'timestamp' in field_names:
            print_ok("Detected Livox-style format (line + timestamp)")
            sensor_type = "livox"
        elif 'ring' in field_names and 't' in field_names:
            print_ok("Detected Ouster-style format (ring + t)")
            sensor_type = "ouster"
        else:
            print_warn(f"Unknown sensor format. Fields: {field_names}")
            sensor_type = "unknown"

        # Calculate frequency
        if len(lidar_msgs) > 1:
            times = [m['time'] for m in lidar_msgs]
            dt = np.diff(times)
            freq = 1.0 / np.mean(dt)
            print(f"\nLiDAR frequency: {freq:.1f} Hz")

            if freq < 5 or freq > 20:
                print_warn(f"Unusual LiDAR frequency (expected 10Hz)")
            else:
                print_ok("LiDAR frequency normal")

        self.results['lidar'] = True
        self.lidar_sensor_type = sensor_type

    def validate_gps(self):
        print_header("4. GPS/FPA Data Validation")

        bag = rosbag.Bag(self.bag_path)

        # Check for different GPS topics
        gps_topics = ['/fixposition/fpa/odometry', '/fixposition/fpa/odomenu', '/odometry/gps']
        gps_data = defaultdict(list)

        for topic, msg, t in bag.read_messages(topics=gps_topics):
            pos = msg.pose.pose.position
            orient = msg.pose.pose.orientation

            gps_data[topic].append({
                'time': msg.header.stamp.to_sec(),
                'frame_id': msg.header.frame_id,
                'pos': [pos.x, pos.y, pos.z],
                'quat': [orient.x, orient.y, orient.z, orient.w]
            })

            if len(gps_data[topic]) > 1000:
                break

        bag.close()

        for topic, data in gps_data.items():
            if len(data) == 0:
                continue

            print(f"\n{Colors.BOLD}Topic: {topic}{Colors.ENDC}")
            print(f"  Messages: {len(data)}")
            print(f"  Frame ID: {data[0]['frame_id']}")

            # Position analysis
            positions = np.array([d['pos'] for d in data])
            pos_mean = np.mean(positions, axis=0)
            pos_std = np.std(positions, axis=0)
            pos_range = np.max(positions, axis=0) - np.min(positions, axis=0)

            print(f"\n  Position statistics:")
            print(f"    First: [{positions[0][0]:.2f}, {positions[0][1]:.2f}, {positions[0][2]:.2f}]")
            print(f"    Mean:  [{pos_mean[0]:.2f}, {pos_mean[1]:.2f}, {pos_mean[2]:.2f}]")
            print(f"    Range: [{pos_range[0]:.2f}, {pos_range[1]:.2f}, {pos_range[2]:.2f}]")

            # Detect coordinate system
            if np.abs(pos_mean[0]) > 1e6:  # ECEF coordinates are in millions
                print_info("  Coordinate system: ECEF (large values)")

                # Calculate approximate lat/lon from ECEF
                x, y, z = pos_mean
                lon = np.arctan2(y, x) * 180 / np.pi
                lat = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                print(f"    Approximate location: lat={lat:.4f}, lon={lon:.4f}")
            else:
                print_info("  Coordinate system: Local (ENU or similar)")

            # Calculate velocity from position changes
            if len(data) > 10:
                times = np.array([d['time'] for d in data])
                dt = np.diff(times)
                dp = np.diff(positions, axis=0)
                velocities = dp / dt[:, np.newaxis]

                vel_norms = np.linalg.norm(velocities, axis=1)
                print(f"\n  Velocity from position (GPS-derived):")
                print(f"    Mean speed: {np.mean(vel_norms):.2f} m/s")
                print(f"    Max speed:  {np.max(vel_norms):.2f} m/s")
                print(f"    Min speed:  {np.min(vel_norms):.2f} m/s")

                if np.max(vel_norms) > 50:
                    print_warn("  Unusually high velocity detected!")
                else:
                    print_ok("  Velocity range reasonable")

            # Orientation analysis
            quats = np.array([d['quat'] for d in data[:100]])
            r = R.from_quat(quats[0])
            euler = r.as_euler('xyz', degrees=True)
            print(f"\n  Initial GPS orientation:")
            print(f"    Roll:  {euler[0]:.2f} deg")
            print(f"    Pitch: {euler[1]:.2f} deg")
            print(f"    Yaw:   {euler[2]:.2f} deg")

            if '/fpa/odometry' in topic:
                self.gps_initial_yaw = euler[2]
                self.gps_ecef_origin = positions[0]

        self.results['gps'] = len(gps_data) > 0

    def validate_time_sync(self):
        print_header("5. Time Synchronization Validation")

        bag = rosbag.Bag(self.bag_path)

        first_times = {}
        topics = ['/imu/data', '/lidar_points', '/fixposition/fpa/odometry']

        for topic, msg, t in bag.read_messages(topics=topics):
            if topic not in first_times:
                first_times[topic] = msg.header.stamp.to_sec()

            if len(first_times) == len(topics):
                break

        bag.close()

        if len(first_times) < 2:
            print_fail("Not enough topics to compare timing")
            self.results['time_sync'] = False
            return

        print("First message timestamps:")
        base_time = min(first_times.values())

        for topic, time in sorted(first_times.items()):
            offset = (time - base_time) * 1000  # Convert to ms
            print(f"  {topic}: +{offset:.1f} ms")

        # Check sync
        max_offset = (max(first_times.values()) - min(first_times.values())) * 1000

        if max_offset > 500:
            print_warn(f"Large time offset between sensors: {max_offset:.1f} ms")
        else:
            print_ok(f"Sensors reasonably synchronized (max offset: {max_offset:.1f} ms)")

        self.results['time_sync'] = max_offset < 1000

    def validate_coordinate_frames(self):
        print_header("6. Coordinate Frame Validation")

        # Compare IMU and GPS yaw
        if hasattr(self, 'imu_initial_yaw') and hasattr(self, 'gps_initial_yaw'):
            yaw_diff = self.imu_initial_yaw - self.gps_initial_yaw
            # Normalize to [-180, 180]
            while yaw_diff > 180:
                yaw_diff -= 360
            while yaw_diff < -180:
                yaw_diff += 360

            print(f"IMU initial yaw: {self.imu_initial_yaw:.2f} deg")
            print(f"GPS initial yaw: {self.gps_initial_yaw:.2f} deg")
            print(f"Yaw difference:  {yaw_diff:.2f} deg")

            if abs(yaw_diff) < 30:
                print_ok("IMU and GPS yaw approximately aligned")
            elif abs(abs(yaw_diff) - 90) < 30:
                print_warn("IMU and GPS yaw differ by ~90 degrees - possible coordinate frame mismatch")
            elif abs(abs(yaw_diff) - 180) < 30:
                print_warn("IMU and GPS yaw differ by ~180 degrees - possible coordinate frame flip")
            else:
                print_warn(f"Significant yaw difference: {yaw_diff:.2f} degrees")
        else:
            print_warn("Cannot compare IMU and GPS yaw (data not available)")

        # Test ECEF to ENU conversion
        if hasattr(self, 'gps_ecef_origin'):
            print(f"\nECEF to ENU conversion test:")
            ecef = self.gps_ecef_origin
            print(f"  ECEF origin: [{ecef[0]:.2f}, {ecef[1]:.2f}, {ecef[2]:.2f}]")

            # Calculate lat/lon
            x, y, z = ecef
            a = 6378137.0  # WGS84 semi-major axis
            f = 1.0 / 298.257223563
            e2 = 2*f - f*f

            lon = np.arctan2(y, x)
            p = np.sqrt(x**2 + y**2)
            lat = np.arctan2(z, p * (1 - e2))

            # Iterate for better lat
            for _ in range(5):
                N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
                lat = np.arctan2(z, p * (1 - e2 * N / (N + 0)))

            print(f"  Latitude:  {np.degrees(lat):.6f} deg")
            print(f"  Longitude: {np.degrees(lon):.6f} deg")

            # Build rotation matrix
            sin_lat, cos_lat = np.sin(lat), np.cos(lat)
            sin_lon, cos_lon = np.sin(lon), np.cos(lon)

            R_ecef_to_enu = np.array([
                [-sin_lon, cos_lon, 0],
                [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
                [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
            ])

            print(f"\n  ENU axes in ECEF:")
            print(f"    East:  [{R_ecef_to_enu[0,0]:.4f}, {R_ecef_to_enu[0,1]:.4f}, {R_ecef_to_enu[0,2]:.4f}]")
            print(f"    North: [{R_ecef_to_enu[1,0]:.4f}, {R_ecef_to_enu[1,1]:.4f}, {R_ecef_to_enu[1,2]:.4f}]")
            print(f"    Up:    [{R_ecef_to_enu[2,0]:.4f}, {R_ecef_to_enu[2,1]:.4f}, {R_ecef_to_enu[2,2]:.4f}]")

            print_ok("ECEF to ENU rotation matrix computed")

        self.results['coordinate_frames'] = True

    def validate_velocity_estimation(self):
        print_header("7. Velocity Estimation Validation")

        bag = rosbag.Bag(self.bag_path)

        # Collect IMU and GPS data for velocity comparison
        imu_data = []
        gps_data = []

        for topic, msg, t in bag.read_messages(topics=['/imu/data', '/fixposition/fpa/odometry']):
            if topic == '/imu/data':
                imu_data.append({
                    'time': msg.header.stamp.to_sec(),
                    'acc': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                })
            elif topic == '/fixposition/fpa/odometry':
                pos = msg.pose.pose.position
                gps_data.append({
                    'time': msg.header.stamp.to_sec(),
                    'pos': [pos.x, pos.y, pos.z]
                })

            if len(imu_data) > 10000 and len(gps_data) > 500:
                break

        bag.close()

        if len(imu_data) < 100 or len(gps_data) < 10:
            print_fail("Insufficient data for velocity validation")
            self.results['velocity'] = False
            return

        print(f"IMU samples: {len(imu_data)}")
        print(f"GPS samples: {len(gps_data)}")

        # GPS-derived velocity
        gps_times = np.array([d['time'] for d in gps_data])
        gps_pos = np.array([d['pos'] for d in gps_data])

        # Convert ECEF to local if needed
        if np.abs(gps_pos[0, 0]) > 1e6:
            print_info("Converting ECEF to local ENU for velocity calculation...")
            origin = gps_pos[0]

            # Build rotation matrix
            x, y, z = origin
            lon = np.arctan2(y, x)
            lat = np.arctan2(z, np.sqrt(x**2 + y**2))

            sin_lat, cos_lat = np.sin(lat), np.cos(lat)
            sin_lon, cos_lon = np.sin(lon), np.cos(lon)

            R_ecef_to_enu = np.array([
                [-sin_lon, cos_lon, 0],
                [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
                [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
            ])

            # Convert to local
            local_pos = np.array([R_ecef_to_enu @ (p - origin) for p in gps_pos])
        else:
            local_pos = gps_pos

        # Calculate GPS velocity
        dt = np.diff(gps_times)
        dp = np.diff(local_pos, axis=0)
        gps_vel = dp / dt[:, np.newaxis]
        gps_speed = np.linalg.norm(gps_vel, axis=1)

        print(f"\nGPS-derived velocity:")
        print(f"  Mean speed:  {np.mean(gps_speed):.2f} m/s ({np.mean(gps_speed)*3.6:.1f} km/h)")
        print(f"  Max speed:   {np.max(gps_speed):.2f} m/s ({np.max(gps_speed)*3.6:.1f} km/h)")
        print(f"  Std speed:   {np.std(gps_speed):.2f} m/s")

        # Find velocity spikes
        speed_threshold = np.mean(gps_speed) + 3 * np.std(gps_speed)
        spikes = np.where(gps_speed > speed_threshold)[0]

        if len(spikes) > 0:
            print_warn(f"  {len(spikes)} velocity spikes detected (>{speed_threshold:.1f} m/s)")
            print(f"  Spike times: {gps_times[spikes[:5]]}")
        else:
            print_ok("  No significant velocity spikes")

        # IMU integration test (simple forward Euler)
        print(f"\nIMU integration test (first 5 seconds):")

        imu_times = np.array([d['time'] for d in imu_data])
        imu_acc = np.array([d['acc'] for d in imu_data])

        # Remove gravity (assuming Z-up)
        gravity = np.array([0, 0, 9.81])
        imu_acc_nograv = imu_acc - gravity

        # Integrate acceleration to get velocity
        dt = np.diff(imu_times)
        integrated_vel = np.zeros((len(imu_data), 3))

        for i in range(1, min(len(imu_data), 1000)):  # First ~5 seconds
            integrated_vel[i] = integrated_vel[i-1] + imu_acc_nograv[i-1] * dt[i-1]

        integrated_speed = np.linalg.norm(integrated_vel, axis=1)

        print(f"  After integration:")
        print(f"    Max velocity: {np.max(integrated_speed[:1000]):.2f} m/s")
        print(f"    Final velocity: {integrated_speed[999]:.2f} m/s")

        if np.max(integrated_speed[:1000]) > 100:
            print_warn("  IMU integration shows drift - this is normal without correction")
        else:
            print_ok("  IMU integration within expected bounds")

        # Velocity direction analysis
        print(f"\nVelocity direction analysis:")
        mean_vel = np.mean(gps_vel, axis=0)
        vel_heading = np.arctan2(mean_vel[1], mean_vel[0]) * 180 / np.pi
        print(f"  Mean velocity vector: [{mean_vel[0]:.2f}, {mean_vel[1]:.2f}, {mean_vel[2]:.2f}] m/s")
        print(f"  Mean heading (ENU): {vel_heading:.2f} deg")

        # Total displacement
        total_disp = local_pos[-1] - local_pos[0]
        disp_heading = np.arctan2(total_disp[1], total_disp[0]) * 180 / np.pi
        print(f"\n  Total displacement: [{total_disp[0]:.2f}, {total_disp[1]:.2f}, {total_disp[2]:.2f}] m")
        print(f"  Displacement heading: {disp_heading:.2f} deg")
        print(f"  Horizontal distance: {np.sqrt(total_disp[0]**2 + total_disp[1]**2):.2f} m")

        self.results['velocity'] = True

    def print_summary(self):
        print_header("Validation Summary")

        all_passed = True
        for test, passed in self.results.items():
            status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}FAIL{Colors.ENDC}"
            print(f"  {test}: {status}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print_ok("All validations passed!")
        else:
            print_warn("Some validations failed - review output above")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 full_system_validator.py <bag_file>")
        sys.exit(1)

    bag_path = sys.argv[1]

    if not os.path.exists(bag_path):
        print(f"Error: Bag file not found: {bag_path}")
        sys.exit(1)

    validator = FullSystemValidator(bag_path)
    validator.run_all_validations()


if __name__ == "__main__":
    main()
