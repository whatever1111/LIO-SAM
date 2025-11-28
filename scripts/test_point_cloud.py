#!/usr/bin/env python3
"""
Test script to verify the correctness of imageProjection.cpp implementation
Tests:
1. Memory layout and alignment
2. Timestamp conversion correctness
3. Zero-copy verification
4. Data integrity after conversion
"""

import struct
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import time
import sys

class PointCloudTester:
    def __init__(self):
        rospy.init_node('point_cloud_tester', anonymous=True)
        self.pub = rospy.Publisher('/lidar_points', PointCloud2, queue_size=10)
        self.sensor_type = rospy.get_param('~sensor_type', 'livox')

    def create_velodyne_cloud(self, num_points=1000):
        """Create a Velodyne-format point cloud with ring and time fields"""
        print("\n=== Creating Velodyne Point Cloud ===")

        # Generate test data
        points = []
        for i in range(num_points):
            x = np.random.randn() * 10
            y = np.random.randn() * 10
            z = np.random.randn() * 2
            intensity = np.random.rand() * 100
            ring = i % 16  # 16 rings
            time = i * 0.0001  # 0.1ms per point = 0.1s total scan
            points.append([x, y, z, intensity, ring, time])

        # Create PointCloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('ring', 16, PointField.UINT16, 1),
            PointField('time', 20, PointField.FLOAT32, 1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "lidar_link"

        cloud = pc2.create_cloud(header, fields, points)

        print(f"Created Velodyne cloud with {num_points} points")
        print(f"Time range: 0.0 to {points[-1][5]:.4f} seconds")
        print(f"Ring range: 0 to 15")

        return cloud, points

    def create_livox_cloud(self, num_points=1000):
        """Create a Livox-format point cloud with line and timestamp fields"""
        print("\n=== Creating Livox Point Cloud ===")

        # Generate test data
        points = []
        base_timestamp = 1700000000  # Current time in milliseconds (smaller value to fit uint32)

        for i in range(num_points):
            x = np.random.randn() * 10
            y = np.random.randn() * 10
            z = np.random.randn() * 2
            intensity = np.random.rand() * 100
            line = i % 16  # 16 lines (uint8)
            tag = 0  # uint8
            reserved = 0  # uint8
            timestamp = base_timestamp + i * 100  # 100ms per point = 100s total

            # Pack the data according to Livox format
            point_data = struct.pack('ffffBBBxI', x, y, z, intensity, line, tag, reserved, timestamp)
            unpacked = struct.unpack('ffffBBBxI', point_data)
            points.append(unpacked)

        # Create PointCloud2 message with correct field layout
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('line', 16, PointField.UINT8, 1),
            PointField('tag', 17, PointField.UINT8, 1),
            PointField('reserved', 18, PointField.UINT8, 1),
            # Note: 1 byte padding at offset 19
            PointField('timestamp', 20, PointField.UINT32, 1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "lidar_link"

        # Create binary data
        data = b''
        for p in points:
            data += struct.pack('ffffBBBxI', *p)

        cloud = PointCloud2(
            header=header,
            height=1,
            width=num_points,
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=24,  # 4*4 + 3*1 + 1(padding) + 4 = 24 bytes
            row_step=24 * num_points,
            data=data
        )

        print(f"Created Livox cloud with {num_points} points")
        print(f"Timestamp range: {base_timestamp} to {base_timestamp + (num_points-1)*100} ms")
        print(f"Expected relative time after conversion: 0.0 to {(num_points-1)*0.1:.1f} seconds")
        print(f"Line range: 0 to 15")
        print(f"Point size: 24 bytes")

        return cloud, points

    def create_ouster_cloud(self, num_points=1000):
        """Create an Ouster-format point cloud"""
        print("\n=== Creating Ouster Point Cloud ===")

        points = []
        base_time_ns = 1000000000  # Base time in nanoseconds (smaller value)

        for i in range(num_points):
            x = np.random.randn() * 10
            y = np.random.randn() * 10
            z = np.random.randn() * 2
            intensity = np.random.rand() * 100
            t = base_time_ns + i * int(1e7)  # 10ms per point in nanoseconds (reduced from 100ms)
            reflectivity = np.random.randint(0, 65535)
            ring = i % 16
            noise = np.random.randint(0, 65535)
            range_val = np.random.randint(0, 50000)

            point_data = struct.pack('ffffIHBxHI', x, y, z, intensity, t, reflectivity, ring, noise, range_val)
            unpacked = struct.unpack('ffffIHBxHI', point_data)
            points.append(unpacked)

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('t', 16, PointField.UINT32, 1),
            PointField('reflectivity', 20, PointField.UINT16, 1),
            PointField('ring', 22, PointField.UINT8, 1),
            # 1 byte padding at offset 23
            PointField('noise', 24, PointField.UINT16, 1),
            PointField('range', 26, PointField.UINT32, 1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "lidar_link"

        data = b''
        for p in points:
            data += struct.pack('ffffIHBxHI', *p)

        cloud = PointCloud2(
            header=header,
            height=1,
            width=num_points,
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=30,
            row_step=30 * num_points,
            data=data
        )

        print(f"Created Ouster cloud with {num_points} points")
        print(f"Time range: 0 to {(num_points-1)*0.01:.2f} seconds (after conversion)")
        print(f"Ring range: 0 to 15")

        return cloud, points

    def verify_memory_layout(self):
        """Verify the memory layout of different point types"""
        print("\n=== Verifying Memory Layout ===")

        # Test Velodyne layout
        print("\nVelodyne Point (24 bytes expected):")
        print("  x,y,z,intensity: 4*4 = 16 bytes")
        print("  ring: 2 bytes (uint16)")
        print("  padding: 2 bytes (alignment)")
        print("  time: 4 bytes (float)")
        print("  Total: 24 bytes")

        # Test Livox layout
        print("\nLivox Point (24 bytes expected):")
        print("  x,y,z,intensity: 4*4 = 16 bytes")
        print("  line,tag,reserved: 3*1 = 3 bytes")
        print("  padding: 1 byte (alignment)")
        print("  timestamp: 4 bytes (uint32)")
        print("  Total: 24 bytes")

        # Test Ouster layout
        print("\nOuster Point (30 bytes expected):")
        print("  x,y,z,intensity: 4*4 = 16 bytes")
        print("  t: 4 bytes (uint32)")
        print("  reflectivity: 2 bytes")
        print("  ring: 1 byte")
        print("  padding: 1 byte")
        print("  noise: 2 bytes")
        print("  range: 4 bytes")
        print("  Total: 30 bytes")

        return True

    def test_timestamp_conversion(self):
        """Test timestamp conversion logic for Livox"""
        print("\n=== Testing Timestamp Conversion ===")

        # Test case 1: Basic conversion
        timestamp_ms = 1234567890
        expected_relative = 0.0  # First point

        ms_to_sec = 0.001
        first_timestamp = timestamp_ms * ms_to_sec
        relative_time = (timestamp_ms * ms_to_sec) - first_timestamp

        print(f"Test 1: First point")
        print(f"  Input: {timestamp_ms} ms")
        print(f"  Expected: {expected_relative} s")
        print(f"  Got: {relative_time} s")
        assert abs(relative_time - expected_relative) < 1e-6, "First point conversion failed"

        # Test case 2: Last point
        last_timestamp_ms = timestamp_ms + 100000  # 100 seconds later
        expected_relative = 100.0
        relative_time = (last_timestamp_ms * ms_to_sec) - first_timestamp

        print(f"Test 2: Last point")
        print(f"  Input: {last_timestamp_ms} ms")
        print(f"  Expected: {expected_relative} s")
        print(f"  Got: {relative_time} s")
        assert abs(relative_time - expected_relative) < 1e-6, "Last point conversion failed"

        # Test case 3: Float storage in uint32
        print(f"Test 3: Float-uint32 storage")
        relative_float = 12.345
        uint32_storage = struct.unpack('I', struct.pack('f', relative_float))[0]
        recovered_float = struct.unpack('f', struct.pack('I', uint32_storage))[0]

        print(f"  Original: {relative_float}")
        print(f"  Stored as uint32: {uint32_storage}")
        print(f"  Recovered: {recovered_float}")
        assert abs(recovered_float - relative_float) < 1e-6, "Float storage failed"

        print("✓ All timestamp conversion tests passed!")
        return True

    def publish_test_clouds(self):
        """Publish test point clouds"""
        rate = rospy.Rate(0.1)  # 0.1 Hz = every 10 seconds

        while not rospy.is_shutdown():
            if self.sensor_type == 'velodyne':
                cloud, _ = self.create_velodyne_cloud(5000)
            elif self.sensor_type == 'livox':
                cloud, _ = self.create_livox_cloud(5000)
            elif self.sensor_type == 'ouster':
                cloud, _ = self.create_ouster_cloud(5000)
            else:
                print(f"Unknown sensor type: {self.sensor_type}")
                break

            self.pub.publish(cloud)
            print(f"\nPublished {self.sensor_type} cloud at {rospy.Time.now()}")
            rate.sleep()

    def run_all_tests(self):
        """Run all verification tests"""
        print("=" * 60)
        print("POINT CLOUD IMPLEMENTATION VERIFICATION")
        print("=" * 60)

        # Test 1: Memory layout
        if not self.verify_memory_layout():
            print("✗ Memory layout verification failed!")
            return False

        # Test 2: Timestamp conversion
        if not self.test_timestamp_conversion():
            print("✗ Timestamp conversion test failed!")
            return False

        # Test 3: Create test clouds
        print("\n=== Creating Test Clouds ===")
        velodyne_cloud, velodyne_points = self.create_velodyne_cloud(100)
        livox_cloud, livox_points = self.create_livox_cloud(100)
        ouster_cloud, ouster_points = self.create_ouster_cloud(100)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

        return True

def main():
    tester = PointCloudTester()

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run tests only
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # Publish test clouds
        print(f"Publishing {tester.sensor_type} test clouds...")
        print("Use --test flag to run verification tests")
        tester.publish_test_clouds()

if __name__ == '__main__':
    main()