#!/usr/bin/env python3
"""
Analyze ROS bag file to understand Livox point cloud timestamp format
and identify the root cause of IMU Preintegration Velocity Anomaly
"""

import rospy
import rosbag
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2, Imu
import sensor_msgs.point_cloud2 as pc2
import sys
import argparse

class BagAnalyzer:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.bag = rosbag.Bag(bag_path, 'r')

        # Topics to analyze
        self.lidar_topic = '/lidar_points'
        self.imu_topic = '/imu/data'

        # Storage for analysis
        self.pc_timestamps = []
        self.imu_timestamps = []
        self.time_sync_issues = []

    def analyze_point_cloud_format(self):
        """Analyze the actual point cloud format in the bag"""
        print("\n" + "="*80)
        print("ANALYZING POINT CLOUD FORMAT")
        print("="*80)

        pc_count = 0
        for topic, msg, t in self.bag.read_messages(topics=[self.lidar_topic]):
            pc_count += 1

            if pc_count > 5:  # Analyze first 5 point clouds
                break

            print(f"\n--- Point Cloud #{pc_count} ---")
            print(f"Bag timestamp: {t.to_sec():.6f}")
            print(f"Header timestamp: {msg.header.stamp.to_sec():.6f}")
            print(f"Time difference: {(t.to_sec() - msg.header.stamp.to_sec())*1000:.3f} ms")

            # Print field information
            print(f"\nFields in point cloud:")
            time_field_info = None
            for field in msg.fields:
                print(f"  {field.name:12} - offset: {field.offset:3}, datatype: {field.datatype}, count: {field.count}")
                if field.name in ['time', 'timestamp', 't']:
                    time_field_info = field

            if not time_field_info:
                print("  WARNING: No time-related field found!")
                continue

            # Extract and analyze timestamps from points
            print(f"\nAnalyzing '{time_field_info.name}' field (datatype={time_field_info.datatype}):")

            # Get raw timestamp values
            timestamps = []
            num_points = msg.width * msg.height
            sample_size = min(100, num_points)  # Sample first 100 points

            for i in range(sample_size):
                offset = i * msg.point_step + time_field_info.offset

                # Read based on datatype
                if time_field_info.datatype == 7:  # FLOAT32
                    value = struct.unpack_from('f', msg.data, offset)[0]
                elif time_field_info.datatype == 6:  # UINT32
                    value = struct.unpack_from('I', msg.data, offset)[0]
                elif time_field_info.datatype == 8:  # FLOAT64
                    value = struct.unpack_from('d', msg.data, offset)[0]
                else:
                    print(f"  Unknown datatype: {time_field_info.datatype}")
                    continue

                timestamps.append(value)

                # Print first few values
                if i < 3 or i == sample_size - 1:
                    print(f"  Point {i:3}: {value}")
                elif i == 3:
                    print(f"  ...")

            if timestamps:
                timestamps = np.array(timestamps)
                self.analyze_timestamp_values(timestamps, msg.header.stamp.to_sec())

    def analyze_timestamp_values(self, timestamps, header_time):
        """Analyze timestamp values to determine format"""
        print(f"\nTimestamp Analysis:")
        print(f"  Count: {len(timestamps)}")
        print(f"  Min: {np.min(timestamps):.6f}")
        print(f"  Max: {np.max(timestamps):.6f}")
        print(f"  Range: {np.max(timestamps) - np.min(timestamps):.6f}")
        print(f"  Mean: {np.mean(timestamps):.6f}")

        # Determine format
        ts_range = np.max(timestamps) - np.min(timestamps)
        ts_min = np.min(timestamps)

        print(f"\nFormat Detection:")

        # Check different possibilities
        if ts_min > 1e12:  # Likely absolute milliseconds (Unix time in ms)
            print(f"  ✓ Detected: ABSOLUTE MILLISECONDS (Unix timestamp)")
            print(f"  First timestamp as date: {ts_min/1000:.3f} sec since epoch")

            # Convert to relative seconds
            first_ts_sec = timestamps[0] / 1000.0
            relative_times = (timestamps / 1000.0) - first_ts_sec
            print(f"\n  Converted to relative seconds:")
            print(f"    First point: {relative_times[0]:.6f} sec")
            print(f"    Last point: {relative_times[-1]:.6f} sec")
            print(f"    Scan duration: {relative_times[-1]:.6f} sec")

            # Check if this matches expected scan duration
            expected_scan_duration = 0.1  # 10Hz typical for Livox
            if abs(relative_times[-1] - expected_scan_duration) > 0.05:
                print(f"  ⚠️  WARNING: Scan duration {relative_times[-1]:.6f} differs from expected {expected_scan_duration}")

        elif ts_min > 1e9 and ts_min < 1e12:  # Likely relative milliseconds
            print(f"  ✓ Detected: RELATIVE MILLISECONDS")
            print(f"  Scan duration: {ts_range:.3f} ms = {ts_range/1000:.6f} sec")

        elif ts_range < 1.0 and ts_min >= 0 and ts_min < 1.0:  # Already relative seconds
            print(f"  ✓ Detected: RELATIVE SECONDS (already normalized)")
            print(f"  Scan duration: {ts_range:.6f} sec")

        elif ts_min < 0 or (ts_min > -1.0 and ts_min < 0):  # Relative seconds with negative offset
            print(f"  ✓ Detected: RELATIVE SECONDS WITH OFFSET")
            print(f"  Range: [{ts_min:.6f}, {np.max(timestamps):.6f}] sec")

        else:
            print(f"  ✗ UNKNOWN FORMAT")
            print(f"  Need manual inspection of values")

        # Check monotonicity
        diffs = np.diff(timestamps)
        if np.all(diffs >= 0):
            print(f"\n  Monotonicity: ✓ Timestamps are increasing")
        else:
            neg_count = np.sum(diffs < 0)
            print(f"\n  Monotonicity: ✗ {neg_count} non-monotonic points found!")

    def analyze_imu_lidar_sync(self):
        """Analyze synchronization between IMU and LiDAR"""
        print("\n" + "="*80)
        print("ANALYZING IMU-LIDAR SYNCHRONIZATION")
        print("="*80)

        # Collect timestamps
        pc_msgs = []
        imu_msgs = []

        for topic, msg, t in self.bag.read_messages(topics=[self.lidar_topic]):
            pc_msgs.append((t.to_sec(), msg.header.stamp.to_sec()))
            if len(pc_msgs) >= 10:
                break

        for topic, msg, t in self.bag.read_messages(topics=[self.imu_topic]):
            imu_msgs.append((t.to_sec(), msg.header.stamp.to_sec()))
            if len(imu_msgs) >= 100:
                break

        if not pc_msgs or not imu_msgs:
            print("ERROR: No messages found")
            return

        print(f"\nFound {len(pc_msgs)} point clouds and {len(imu_msgs)} IMU messages")

        # Analyze timing
        print(f"\nPoint Cloud timing:")
        for i, (bag_t, header_t) in enumerate(pc_msgs[:3]):
            print(f"  PC {i}: bag={bag_t:.6f}, header={header_t:.6f}, diff={(bag_t-header_t)*1000:.3f}ms")

        print(f"\nIMU timing:")
        for i, (bag_t, header_t) in enumerate(imu_msgs[:3]):
            print(f"  IMU {i}: bag={bag_t:.6f}, header={header_t:.6f}, diff={(bag_t-header_t)*1000:.3f}ms")

        # Check IMU frequency
        if len(imu_msgs) > 1:
            imu_intervals = np.diff([t[1] for t in imu_msgs])
            imu_freq = 1.0 / np.mean(imu_intervals)
            print(f"\nIMU frequency: {imu_freq:.1f} Hz (expected: 100-500 Hz)")

            if imu_freq < 50:
                print(f"  ⚠️  WARNING: IMU frequency is too low!")

        # Check PC frequency
        if len(pc_msgs) > 1:
            pc_intervals = np.diff([t[1] for t in pc_msgs])
            pc_freq = 1.0 / np.mean(pc_intervals)
            print(f"LiDAR frequency: {pc_freq:.1f} Hz (expected: 10 Hz)")

    def check_velocity_anomaly(self):
        """Check for velocity anomalies in the data"""
        print("\n" + "="*80)
        print("CHECKING FOR VELOCITY ANOMALIES")
        print("="*80)

        # Look for IMU data with high accelerations
        max_accel = 0
        anomaly_count = 0

        for topic, msg, t in self.bag.read_messages(topics=[self.imu_topic]):
            accel = np.array([msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z])
            accel_norm = np.linalg.norm(accel)

            if accel_norm > max_accel:
                max_accel = accel_norm

            # Check for anomalies (acceleration > 20 m/s^2 excluding gravity)
            accel_no_gravity = accel_norm - 9.81
            if abs(accel_no_gravity) > 20:
                anomaly_count += 1
                if anomaly_count <= 3:
                    print(f"  Anomaly at {t.to_sec():.6f}: |a| = {accel_norm:.2f} m/s^2")

        print(f"\nMax acceleration: {max_accel:.2f} m/s^2")
        print(f"Anomalous IMU readings: {anomaly_count}")

    def suggest_fix(self):
        """Suggest fixes based on analysis"""
        print("\n" + "="*80)
        print("SUGGESTED FIXES")
        print("="*80)

        print("""
Based on the analysis, here are potential fixes:

1. If timestamps are in ABSOLUTE MILLISECONDS:
   - Convert to relative seconds: (timestamp_ms - first_timestamp_ms) / 1000.0
   - Store as float32 in the same field

2. If timestamps are in RELATIVE MILLISECONDS:
   - Convert to seconds: timestamp_ms / 1000.0
   - Ensure proper type casting

3. If timestamps are already in RELATIVE SECONDS:
   - Remove any conversion in imageProjection.cpp
   - Use timestamps directly

4. For IMU synchronization:
   - Ensure point cloud header timestamp matches scan start time
   - Check that IMU messages cover the entire scan duration
   - Verify IMU frequency is sufficient (>100 Hz)

5. Common issues to check:
   - Endianness of timestamp field
   - Proper memory alignment in struct
   - Race conditions in parallel processing
   - Clock synchronization between sensors
""")

    def run_analysis(self):
        """Run complete analysis"""
        try:
            print(f"\nAnalyzing bag file: {self.bag_path}")
            print("="*80)

            # Run all analyses
            self.analyze_point_cloud_format()
            self.analyze_imu_lidar_sync()
            self.check_velocity_anomaly()
            self.suggest_fix()

        finally:
            self.bag.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze ROS bag for timestamp issues')
    parser.add_argument('bag_file', help='Path to ROS bag file')
    parser.add_argument('--lidar-topic', default='/lidar_points', help='LiDAR topic name')
    parser.add_argument('--imu-topic', default='/imu/data', help='IMU topic name')

    args = parser.parse_args()

    analyzer = BagAnalyzer(args.bag_file)
    if args.lidar_topic:
        analyzer.lidar_topic = args.lidar_topic
    if args.imu_topic:
        analyzer.imu_topic = args.imu_topic

    analyzer.run_analysis()

if __name__ == '__main__':
    # If no arguments provided, use default bag file if it exists
    if len(sys.argv) == 1:
        default_bag = '/root/autodl-tmp/info_fixed.bag'
        print(f"No bag file specified, trying default: {default_bag}")
        analyzer = BagAnalyzer(default_bag)
        analyzer.run_analysis()
    else:
        main()