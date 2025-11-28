#!/usr/bin/env python
import rosbag
import struct
import numpy as np

def analyze_timestamp_distribution(bag_path):
    """Analyze timestamp distribution within point clouds"""
    bag = rosbag.Bag(bag_path)

    print("="*70)
    print("POINT CLOUD TIMESTAMP DISTRIBUTION ANALYSIS")
    print("="*70)

    msg_count = 0
    for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
        if msg_count >= 3:  # Analyze first 3 messages
            break

        msg_count += 1
        print(f"\n\nMessage {msg_count}:")
        print(f"  Header time: {msg.header.stamp.to_sec():.6f}")

        # Extract timestamps
        data = msg.data
        point_step = msg.point_step
        num_points = msg.width * msg.height

        # Find timestamp field
        timestamp_field = None
        for field in msg.fields:
            if field.name == 'timestamp':
                timestamp_field = field
                break

        if timestamp_field:
            timestamps = []

            # Read all timestamps
            for i in range(min(num_points, 100000)):  # Limit to avoid memory issues
                offset = i * point_step + timestamp_field.offset
                if offset + 8 <= len(data):
                    ts = struct.unpack('d', data[offset:offset+8])[0]
                    timestamps.append(ts)

            timestamps = np.array(timestamps)

            print(f"\n  Total points analyzed: {len(timestamps)}")
            print(f"  Raw timestamp range: [{np.min(timestamps):.6f}, {np.max(timestamps):.6f}]")
            print(f"  Raw timestamp span: {np.max(timestamps) - np.min(timestamps):.6f}")

            # Convert to seconds
            timestamps_sec = timestamps / 1000.0
            first_ts_sec = timestamps_sec[0]

            print(f"\n  After /1000 (to seconds):")
            print(f"    Range: [{np.min(timestamps_sec):.6f}, {np.max(timestamps_sec):.6f}]")
            print(f"    First point: {first_ts_sec:.6f}")

            # Calculate relative times
            relative_times = timestamps_sec - first_ts_sec

            print(f"\n  Relative times (from first point):")
            print(f"    Min: {np.min(relative_times):.6f}")
            print(f"    Max: {np.max(relative_times):.6f}")
            print(f"    Mean: {np.mean(relative_times):.6f}")

            # Sample some points to see the pattern
            print(f"\n  Sample relative times (every 10000th point):")
            for i in range(0, len(relative_times), 10000):
                print(f"    Point {i:5d}: {relative_times[i]:.6f} sec")

            # Check if timestamps are monotonic
            is_monotonic = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            print(f"\n  Timestamps monotonic: {is_monotonic}")

            # Find min and max timestamps and their positions
            min_idx = np.argmin(timestamps)
            max_idx = np.argmax(timestamps)
            print(f"\n  Min timestamp at index: {min_idx}")
            print(f"  Max timestamp at index: {max_idx}")

            # Check for timestamp jumps
            diffs = np.diff(relative_times)
            print(f"\n  Time differences between consecutive points:")
            print(f"    Mean: {np.mean(diffs)*1000:.6f} ms")
            print(f"    Std:  {np.std(diffs)*1000:.6f} ms")
            print(f"    Max jump: {np.max(np.abs(diffs))*1000:.6f} ms")

    bag.close()

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"
    analyze_timestamp_distribution(bag_path)