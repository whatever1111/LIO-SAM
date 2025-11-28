#!/usr/bin/env python
import rosbag
import struct
import numpy as np

def check_pointcloud_timestamps(bag_path):
    """Check the timestamp field values in point cloud messages"""
    bag = rosbag.Bag(bag_path)

    print("Checking point cloud timestamp fields...")
    print("="*60)

    # Check first few point clouds
    count = 0
    for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
        if count >= 3:  # Check first 3 messages
            break

        count += 1
        print(f"\nMessage {count}:")
        print(f"  Header time: {msg.header.stamp.to_sec():.6f}")
        print(f"  Bag time: {t.to_sec():.6f}")

        # Extract raw data
        data = msg.data
        point_step = msg.point_step

        # Check if timestamp field exists
        timestamp_field = None
        for field in msg.fields:
            if field.name == 'timestamp':
                timestamp_field = field
                break

        if timestamp_field:
            print(f"  Timestamp field found at offset {timestamp_field.offset}")

            # Read first and last point timestamps
            if len(data) >= point_step:
                # First point
                offset = timestamp_field.offset
                first_timestamp = struct.unpack('d', data[offset:offset+8])[0]

                # Last point
                last_offset = (msg.width * msg.height - 1) * point_step + timestamp_field.offset
                if last_offset + 8 <= len(data):
                    last_timestamp = struct.unpack('d', data[last_offset:last_offset+8])[0]
                else:
                    last_timestamp = 0

                # Sample a few points
                num_points = msg.width * msg.height
                sample_indices = [0, num_points//4, num_points//2, 3*num_points//4, num_points-1]

                print(f"  Point timestamps (sampled):")
                for idx in sample_indices:
                    if idx < num_points:
                        offset = idx * point_step + timestamp_field.offset
                        if offset + 8 <= len(data):
                            ts = struct.unpack('d', data[offset:offset+8])[0]
                            print(f"    Point {idx:5d}: {ts:.6f}")

                print(f"  Time range in cloud: {first_timestamp:.6f} to {last_timestamp:.6f}")
                print(f"  Duration: {(last_timestamp - first_timestamp)*1000:.2f} ms")

                # Check if timestamps are relative or absolute
                if first_timestamp > 1e9:
                    print("  -> Timestamps appear to be ABSOLUTE (Unix time)")
                elif first_timestamp < 1:
                    print("  -> Timestamps appear to be RELATIVE (seconds from scan start)")
                else:
                    print("  -> Timestamps format UNCLEAR")
        else:
            print("  No timestamp field found in point cloud!")
            print("  Available fields:", [f.name for f in msg.fields])

    bag.close()

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"
    check_pointcloud_timestamps(bag_path)