#!/usr/bin/env python3
"""
Diagnostic script to identify timestamp synchronization issues causing IMU Preintegration Velocity Anomaly
This will help us understand the actual Livox timestamp format and how it relates to IMU data
"""

import rospy
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
import struct
import numpy as np
from collections import deque
import sys

class TimestampDiagnostic:
    def __init__(self):
        rospy.init_node('timestamp_diagnostic', anonymous=True)

        # Subscribers
        self.pc_sub = rospy.Subscriber('/lidar_points', PointCloud2, self.pc_callback, queue_size=1)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=100)
        self.odom_sub = rospy.Subscriber('/lio_sam/mapping/odometry_incremental', Odometry,
                                        self.odom_callback, queue_size=10)

        # Data storage
        self.pc_data = None
        self.imu_queue = deque(maxlen=1000)
        self.odom_queue = deque(maxlen=100)

        # Analysis flags
        self.analyzed = False
        self.pc_count = 0

        # Timing statistics
        self.timing_stats = {
            'pc_header_times': [],
            'pc_first_point_times': [],
            'pc_last_point_times': [],
            'scan_durations': [],
            'imu_times': [],
            'time_differences': []
        }

    def pc_callback(self, msg):
        """Analyze point cloud timing in detail"""
        self.pc_count += 1

        print(f"\n{'='*80}")
        print(f"POINT CLOUD #{self.pc_count} ANALYSIS")
        print(f"{'='*80}")

        # Header timestamp
        header_time = msg.header.stamp.to_sec()
        print(f"\n1. HEADER TIMESTAMP:")
        print(f"   ROS Time: {msg.header.stamp.to_sec():.6f} sec")
        print(f"   Frame ID: {msg.header.frame_id}")

        # Field information
        print(f"\n2. POINT CLOUD FIELDS:")
        timestamp_field = None
        for field in msg.fields:
            print(f"   - {field.name}: offset={field.offset}, datatype={field.datatype}, count={field.count}")
            if field.name in ['time', 'timestamp', 't']:
                timestamp_field = field

        if not timestamp_field:
            print("   WARNING: No timestamp field found!")
            return

        # Extract points to analyze timestamps
        print(f"\n3. ANALYZING TIMESTAMPS (Field: '{timestamp_field.name}'):")
        print(f"   Total points: {msg.width * msg.height}")
        print(f"   Point step: {msg.point_step} bytes")

        # Read first, middle, and last points
        points_to_check = min(10, msg.width)  # Check first 10 points
        timestamps = []

        for i in range(msg.width):
            offset = i * msg.point_step + timestamp_field.offset

            # Read timestamp based on datatype
            if timestamp_field.datatype == 7:  # FLOAT32
                timestamp = struct.unpack_from('f', msg.data, offset)[0]
            elif timestamp_field.datatype == 6:  # UINT32
                timestamp = struct.unpack_from('I', msg.data, offset)[0]
            elif timestamp_field.datatype == 8:  # FLOAT64
                timestamp = struct.unpack_from('d', msg.data, offset)[0]
            else:
                print(f"   Unknown timestamp datatype: {timestamp_field.datatype}")
                continue

            timestamps.append(timestamp)

            if i < 5 or i == msg.width//2 or i >= msg.width - 5:
                if i < 5:
                    label = f"Point {i}"
                elif i == msg.width//2:
                    label = f"Middle point ({i})"
                else:
                    label = f"Point {i}"
                print(f"   {label}: raw_value={timestamp}")

        if timestamps:
            timestamps = np.array(timestamps)

            # Analyze timestamp patterns
            print(f"\n4. TIMESTAMP STATISTICS:")
            print(f"   Min: {np.min(timestamps)}")
            print(f"   Max: {np.max(timestamps)}")
            print(f"   Range: {np.max(timestamps) - np.min(timestamps)}")
            print(f"   Mean: {np.mean(timestamps):.6f}")
            print(f"   Std Dev: {np.std(timestamps):.6f}")

            # Determine timestamp format
            print(f"\n5. TIMESTAMP FORMAT DETECTION:")
            ts_range = np.max(timestamps) - np.min(timestamps)

            # Check if values are likely milliseconds
            if np.min(timestamps) > 1e9:
                print(f"   FORMAT: Likely ABSOLUTE MILLISECONDS (values > 1e9)")
                print(f"   Scan duration if ms: {ts_range:.3f} ms = {ts_range/1000:.6f} seconds")

                # Convert to relative seconds
                first_ts_sec = timestamps[0] / 1000.0
                relative_times = (timestamps / 1000.0) - first_ts_sec
                print(f"\n   After conversion to relative seconds:")
                print(f"   First point: {relative_times[0]:.6f} sec")
                print(f"   Last point: {relative_times[-1]:.6f} sec")

            # Check if values are likely relative seconds
            elif ts_range < 1.0 and np.min(timestamps) >= 0:
                print(f"   FORMAT: Likely RELATIVE SECONDS (range < 1.0)")
                print(f"   Scan duration: {ts_range:.6f} seconds")

            # Check if values might be nanoseconds
            elif np.min(timestamps) > 1e15:
                print(f"   FORMAT: Likely NANOSECONDS")
                print(f"   Scan duration if ns: {ts_range/1e9:.6f} seconds")

            else:
                print(f"   FORMAT: UNKNOWN - needs further analysis")

            # Check for monotonicity
            diffs = np.diff(timestamps)
            if np.all(diffs >= 0):
                print(f"\n6. MONOTONICITY: ✓ Timestamps are monotonically increasing")
            else:
                print(f"\n6. MONOTONICITY: ✗ WARNING - Timestamps are NOT monotonic!")
                print(f"   Negative differences found: {np.sum(diffs < 0)}")

        # Compare with IMU data
        self.compare_with_imu(header_time, timestamps if timestamps else [])

    def imu_callback(self, msg):
        """Store IMU data for timing comparison"""
        self.imu_queue.append({
            'time': msg.header.stamp.to_sec(),
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        })

    def odom_callback(self, msg):
        """Store odometry for velocity checking"""
        self.odom_queue.append({
            'time': msg.header.stamp.to_sec(),
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        })

    def compare_with_imu(self, pc_header_time, pc_timestamps):
        """Compare point cloud timing with IMU data"""
        if not self.imu_queue:
            print("\n7. IMU COMPARISON: No IMU data available")
            return

        print(f"\n7. IMU-LIDAR SYNCHRONIZATION:")

        # Find IMU messages around the point cloud time
        imu_times = [imu['time'] for imu in self.imu_queue]

        # Find closest IMU messages
        closest_before = None
        closest_after = None

        for imu_time in imu_times:
            if imu_time <= pc_header_time:
                closest_before = imu_time
            elif closest_after is None:
                closest_after = imu_time
                break

        if closest_before:
            print(f"   IMU before PC: {closest_before:.6f} (Δt = {pc_header_time - closest_before:.6f} sec)")
        if closest_after:
            print(f"   IMU after PC:  {closest_after:.6f} (Δt = {closest_after - pc_header_time:.6f} sec)")

        # Check IMU coverage for scan duration
        if pc_timestamps:
            scan_duration = np.max(pc_timestamps) - np.min(pc_timestamps)

            # Determine the actual scan time window
            if np.min(pc_timestamps) > 1e9:  # Milliseconds
                scan_duration_sec = scan_duration / 1000.0
            elif np.min(pc_timestamps) > 1e15:  # Nanoseconds
                scan_duration_sec = scan_duration / 1e9
            else:  # Already in seconds
                scan_duration_sec = scan_duration

            scan_end_time = pc_header_time + scan_duration_sec

            print(f"\n   Scan duration: {scan_duration_sec:.6f} seconds")
            print(f"   Scan window: [{pc_header_time:.6f}, {scan_end_time:.6f}]")

            # Count IMU messages during scan
            imu_during_scan = sum(1 for t in imu_times
                                if pc_header_time <= t <= scan_end_time)
            print(f"   IMU messages during scan: {imu_during_scan}")

            if imu_during_scan < 10:
                print(f"   ⚠️  WARNING: Few IMU messages during scan (expected ~100 for 10Hz scan)")

        # Check for velocity anomalies in odometry
        if self.odom_queue:
            print(f"\n8. VELOCITY CHECK:")
            recent_odom = list(self.odom_queue)[-5:]  # Last 5 odometry messages
            for odom in recent_odom:
                vel = odom['velocity']
                vel_norm = np.linalg.norm(vel)
                print(f"   Time {odom['time']:.3f}: |v| = {vel_norm:.3f} m/s "
                      f"(vx={vel[0]:.2f}, vy={vel[1]:.2f}, vz={vel[2]:.2f})")

                if vel_norm > 10.0:
                    print(f"   ⚠️  VELOCITY ANOMALY DETECTED!")

    def run_diagnostic(self):
        """Run the diagnostic for a period of time"""
        print("\n" + "="*80)
        print("TIMESTAMP SYNCHRONIZATION DIAGNOSTIC")
        print("="*80)
        print("\nMonitoring point clouds and IMU data...")
        print("Press Ctrl+C to stop\n")

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n\nDiagnostic complete.")
            self.print_summary()

    def print_summary(self):
        """Print summary of findings"""
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)

        print(f"\nTotal point clouds analyzed: {self.pc_count}")

        if self.timing_stats['scan_durations']:
            print(f"\nScan duration statistics:")
            durations = self.timing_stats['scan_durations']
            print(f"  Mean: {np.mean(durations):.6f} sec")
            print(f"  Std:  {np.std(durations):.6f} sec")

        print("\nRECOMMENDATIONS:")
        print("1. Check if Livox timestamps are in milliseconds or already in relative seconds")
        print("2. Verify IMU frequency matches expected rate (typically 100-500 Hz)")
        print("3. Ensure point cloud header timestamp is synchronized with system time")
        print("4. Check if scan duration matches expected LiDAR rotation period")

if __name__ == '__main__':
    diagnostic = TimestampDiagnostic()
    diagnostic.run_diagnostic()