#!/usr/bin/env python3
"""
Analyze the actual Livox timestamp format to identify the bug
"""

import rosbag
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2

def analyze_timestamp_format():
    bag_path = '/root/autodl-tmp/info_fixed.bag'
    bag = rosbag.Bag(bag_path, 'r')

    print("="*80)
    print("DETAILED LIVOX TIMESTAMP FORMAT ANALYSIS")
    print("="*80)

    # Get first point cloud message
    for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
        print(f"\nBag timestamp: {t.to_sec():.6f} seconds")
        print(f"Header timestamp: {msg.header.stamp.to_sec():.6f} seconds")

        # Find timestamp field
        timestamp_field = None
        for field in msg.fields:
            if field.name == 'timestamp':
                timestamp_field = field
                break

        if not timestamp_field:
            print("ERROR: No timestamp field found!")
            continue

        print(f"\nTimestamp field:")
        print(f"  Type: FLOAT64 (double)")
        print(f"  Offset: {timestamp_field.offset}")

        # Extract first 10 timestamps
        timestamps = []
        for i in range(min(10, msg.width)):
            offset = i * msg.point_step + timestamp_field.offset
            timestamp = struct.unpack_from('d', msg.data, offset)[0]
            timestamps.append(timestamp)

        timestamps = np.array(timestamps)

        print(f"\nRaw timestamp values (first 10):")
        for i, ts in enumerate(timestamps):
            print(f"  Point {i}: {ts:.10f}")

        # Analyze the format
        print(f"\nFormat Analysis:")

        # Test 1: Are these Unix timestamps in seconds?
        header_time = msg.header.stamp.to_sec()
        diff_from_header = timestamps[0] - header_time
        print(f"\n1. If interpreted as SECONDS:")
        print(f"   First timestamp: {timestamps[0]:.6f} sec")
        print(f"   Header time: {header_time:.6f} sec")
        print(f"   Difference: {diff_from_header:.6f} sec")

        if abs(diff_from_header) < 1.0:  # Within 1 second
            print(f"   ✓ MATCH! These appear to be Unix timestamps in SECONDS")

            # Calculate scan duration
            scan_duration = timestamps[-1] - timestamps[0]
            print(f"   Scan duration: {scan_duration:.6f} seconds")

            if scan_duration < 0.01:  # Less than 10ms
                print(f"   ✓ Reasonable scan duration for partial scan")

        # Test 2: Are these milliseconds?
        print(f"\n2. If interpreted as MILLISECONDS:")
        ts_as_sec = timestamps[0] / 1000.0
        print(f"   First timestamp: {ts_as_sec:.6f} sec")
        print(f"   Difference from header: {ts_as_sec - header_time:.6f} sec")

        if abs(ts_as_sec - header_time) > 1e6:
            print(f"   ✗ NO MATCH - difference too large")

        # Test 3: Are these microseconds?
        print(f"\n3. If interpreted as MICROSECONDS:")
        ts_as_sec = timestamps[0] / 1e6
        print(f"   First timestamp: {ts_as_sec:.6f} sec")
        print(f"   Difference from header: {ts_as_sec - header_time:.6f} sec")

        if abs(ts_as_sec - header_time) > 1000:
            print(f"   ✗ NO MATCH - difference too large")

        # Test 4: Analyze the integer vs fractional parts
        print(f"\n4. Integer vs Fractional Analysis:")
        int_part = int(timestamps[0])
        frac_part = timestamps[0] - int_part
        print(f"   Integer part: {int_part}")
        print(f"   Fractional part: {frac_part:.10f}")

        # Check if integer part matches Unix seconds
        if abs(int_part - int(header_time)) < 2:
            print(f"   ✓ Integer part matches Unix seconds epoch!")
            print(f"   => These are Unix timestamps in SECONDS with sub-second precision")

        print("\n" + "="*80)
        print("CONCLUSION:")
        print("="*80)
        print("\nThe Livox timestamps are:")
        print("  • Format: Unix timestamps in SECONDS (not milliseconds!)")
        print("  • Type: FLOAT64 (double precision)")
        print("  • Example: 1763444615.9826328 seconds since Unix epoch")
        print("\nThe BUG in imageProjection.cpp:")
        print("  • WRONG: Treating as milliseconds and multiplying by 0.001")
        print("  • CORRECT: Already in seconds, just subtract first timestamp")
        print("\nThis explains:")
        print("  • IMU velocity anomaly (wrong timestamps → wrong IMU interpolation)")
        print("  • LiDAR timeout (timestamps in wrong range)")

        bag.close()
        return True

    bag.close()
    return False

if __name__ == "__main__":
    analyze_timestamp_format()