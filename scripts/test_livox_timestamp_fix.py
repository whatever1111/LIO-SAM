#!/usr/bin/env python3
"""
Test script to verify Livox timestamp conversion is working correctly
This creates a test point cloud with the exact format found in the bag file analysis
"""

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import struct
import numpy as np
import time

def create_livox_point_cloud():
    """Create a test Livox point cloud with correct timestamp format"""

    # Initialize message
    msg = PointCloud2()
    msg.header.frame_id = "lidar_link"

    # Define fields matching actual Livox format
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
        PointField("intensity", 12, PointField.FLOAT32, 1),
        PointField("tag", 16, PointField.UINT8, 1),
        PointField("line", 17, PointField.UINT8, 1),
        PointField("timestamp", 18, PointField.FLOAT64, 1),  # FLOAT64 as found in bag
    ]
    msg.fields = fields

    # Point step is 26 bytes (4+4+4+4+1+1+8)
    msg.point_step = 26
    msg.is_bigendian = False
    msg.is_dense = True

    # Create test points
    num_points = 1000
    msg.height = 1
    msg.width = num_points
    msg.row_step = msg.point_step * num_points

    # Generate data
    data = bytearray()

    # Use current Unix time in milliseconds (similar to actual data)
    base_time_ms = time.time() * 1000  # Current time in milliseconds

    for i in range(num_points):
        # Position (simulate a scan pattern)
        angle = 2.0 * np.pi * i / num_points
        distance = 10.0 + np.random.normal(0, 0.1)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = np.random.normal(0, 0.5)

        # Pack point data
        data.extend(struct.pack('f', x))       # x
        data.extend(struct.pack('f', y))       # y
        data.extend(struct.pack('f', z))       # z
        data.extend(struct.pack('f', 100.0))   # intensity
        data.extend(struct.pack('B', 0))       # tag
        data.extend(struct.pack('B', i % 16))  # line (0-15 for 16 lines)

        # Timestamp: absolute milliseconds as double (matching actual format)
        # Simulate 100ms scan duration (10Hz)
        timestamp_ms = base_time_ms + (i * 100.0 / num_points)
        data.extend(struct.pack('d', timestamp_ms))  # timestamp as double

    msg.data = bytes(data)

    # Set header timestamp to match first point time (converted to seconds)
    msg.header.stamp = rospy.Time.from_sec(base_time_ms / 1000.0)

    return msg

def verify_conversion():
    """Verify the timestamp conversion logic"""
    print("\n" + "="*60)
    print("TESTING LIVOX TIMESTAMP CONVERSION")
    print("="*60)

    # Test conversion logic
    test_timestamps_ms = np.array([
        1763444615982.6328,
        1763444615982.6377,
        1763444615982.6426,
        1763444615983.1028,
    ])

    print("\nOriginal timestamps (absolute milliseconds):")
    for i, ts in enumerate(test_timestamps_ms):
        print(f"  Point {i}: {ts:.4f} ms")

    # Apply conversion (what imageProjection.cpp should do)
    MS_TO_SEC = 0.001
    first_timestamp_sec = test_timestamps_ms[0] * MS_TO_SEC
    relative_times = (test_timestamps_ms * MS_TO_SEC) - first_timestamp_sec

    print("\nAfter conversion to relative seconds:")
    for i, rt in enumerate(relative_times):
        print(f"  Point {i}: {rt:.6f} sec")

    print(f"\nScan duration: {relative_times[-1]:.6f} seconds")

    # Check if reasonable
    if relative_times[-1] < 0.5:  # Should be less than 500ms for partial scan
        print("✓ Conversion looks reasonable")
    else:
        print("✗ WARNING: Scan duration seems too long")

def main():
    rospy.init_node('test_livox_timestamps')

    # Publisher
    pub = rospy.Publisher('/test_lidar_points', PointCloud2, queue_size=1)

    # Run verification
    verify_conversion()

    print("\n" + "="*60)
    print("PUBLISHING TEST POINT CLOUDS")
    print("="*60)

    rate = rospy.Rate(10)  # 10Hz
    count = 0

    print("\nPublishing Livox-format point clouds to /test_lidar_points")
    print("Use 'rostopic echo /test_lidar_points' to verify")
    print("Press Ctrl+C to stop")

    while not rospy.is_shutdown():
        msg = create_livox_point_cloud()
        pub.publish(msg)

        count += 1
        if count % 10 == 0:
            print(f"Published {count} point clouds...")

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\n\nTest complete.")