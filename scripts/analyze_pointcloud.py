#!/usr/bin/env python3
"""
Analyze point cloud fields to understand the actual data format
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sys

class PointCloudAnalyzer:
    def __init__(self):
        rospy.init_node('pc_analyzer', anonymous=True)
        self.sub = rospy.Subscriber('/lidar_points', PointCloud2, self.callback, queue_size=1)
        self.analyzed = False

    def callback(self, msg):
        if self.analyzed:
            return

        print("\n=== Point Cloud Analysis ===")
        print(f"Header stamp: {msg.header.stamp.to_sec()}")
        print(f"Frame ID: {msg.header.frame_id}")
        print(f"Width: {msg.width}, Height: {msg.height}")
        print(f"Point step: {msg.point_step} bytes")
        print(f"Is dense: {msg.is_dense}")
        print(f"Is bigendian: {msg.is_bigendian}")

        print("\n=== Fields ===")
        for field in msg.fields:
            print(f"  {field.name}:")
            print(f"    offset: {field.offset}")
            print(f"    datatype: {field.datatype} ({self.get_datatype_name(field.datatype)})")
            print(f"    count: {field.count}")

        # Analyze first few points
        print("\n=== First Point Data (if available) ===")
        if msg.width > 0:
            import struct
            # Read first point
            data = msg.data[:msg.point_step]
            offset = 0
            for field in msg.fields:
                if field.name in ['x', 'y', 'z', 'intensity', 'time', 'timestamp', 'line', 'ring']:
                    value = self.read_field(data, field)
                    print(f"  {field.name}: {value}")

        self.analyzed = True
        rospy.signal_shutdown("Analysis complete")

    def get_datatype_name(self, datatype):
        types = {
            1: "INT8", 2: "UINT8", 3: "INT16", 4: "UINT16",
            5: "INT32", 6: "UINT32", 7: "FLOAT32", 8: "FLOAT64"
        }
        return types.get(datatype, "UNKNOWN")

    def read_field(self, data, field):
        offset = field.offset
        if field.datatype == 7:  # FLOAT32
            return struct.unpack_from('f', data, offset)[0]
        elif field.datatype == 6:  # UINT32
            return struct.unpack_from('I', data, offset)[0]
        elif field.datatype == 4:  # UINT16
            return struct.unpack_from('H', data, offset)[0]
        elif field.datatype == 2:  # UINT8
            return struct.unpack_from('B', data, offset)[0]
        else:
            return None

if __name__ == '__main__':
    analyzer = PointCloudAnalyzer()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass