#!/usr/bin/env python3
"""
分析原始 bag 中的 LiDAR 数据格式
"""

import rosbag
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

bag_file = '/root/autodl-tmp/info_fixed.bag'
topic = '/lidar_points'

print("="*70)
print("LiDAR Data Analysis")
print("="*70)

bag = rosbag.Bag(bag_file, 'r')

count = 0
for topic_name, msg, t in bag.read_messages(topics=[topic]):
    count += 1

    if count <= 3:
        print(f"\n--- Frame {count} ---")
        print(f"Header timestamp: {msg.header.stamp.to_sec():.6f}")
        print(f"Frame ID: {msg.header.frame_id}")
        print(f"Height: {msg.height}, Width: {msg.width}")
        print(f"Point step: {msg.point_step}, Row step: {msg.row_step}")
        print(f"Is dense: {msg.is_dense}")
        print(f"Is bigendian: {msg.is_bigendian}")

        print("\nFields:")
        for field in msg.fields:
            print(f"  {field.name}: offset={field.offset}, datatype={field.datatype}, count={field.count}")

        # 读取点云数据
        points = list(pc2.read_points(msg, skip_nans=False))

        if points:
            print(f"\nTotal points: {len(points)}")
            print(f"Point structure (first point): {points[0]}")
            print(f"Number of fields per point: {len(points[0])}")

            # 分析各字段
            field_names = [f.name for f in msg.fields]
            print(f"\nField names: {field_names}")

            # 检查 timestamp 字段
            if 'timestamp' in field_names:
                timestamps = [p[field_names.index('timestamp')] for p in points[:100]]
                print(f"\nTimestamp analysis (first 100 points):")
                print(f"  Min: {min(timestamps)}")
                print(f"  Max: {max(timestamps)}")
                print(f"  Range: {max(timestamps) - min(timestamps)}")
                print(f"  First 5: {timestamps[:5]}")

                # 判断单位
                if max(timestamps) > 1e9:
                    print(f"  Unit: Likely NANOSECONDS")
                elif max(timestamps) > 1e6:
                    print(f"  Unit: Likely MILLISECONDS")
                elif max(timestamps) > 1e3:
                    print(f"  Unit: Likely SECONDS (absolute)")
                else:
                    print(f"  Unit: Likely SECONDS (relative)")

            # 检查 ring 字段
            if 'ring' in field_names:
                rings = [p[field_names.index('ring')] for p in points]
                unique_rings = sorted(set(rings))
                print(f"\nRing analysis:")
                print(f"  Unique rings: {len(unique_rings)}")
                print(f"  Ring range: {min(unique_rings)} - {max(unique_rings)}")

            # 检查 intensity
            if 'intensity' in field_names:
                intensities = [p[field_names.index('intensity')] for p in points[:100]]
                print(f"\nIntensity analysis (first 100):")
                print(f"  Min: {min(intensities)}, Max: {max(intensities)}")

            # XYZ 范围
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            print(f"\nXYZ range:")
            print(f"  X: [{min(xs):.2f}, {max(xs):.2f}]")
            print(f"  Y: [{min(ys):.2f}, {max(ys):.2f}]")
            print(f"  Z: [{min(zs):.2f}, {max(zs):.2f}]")

    if count >= 3:
        break

bag.close()

print("\n" + "="*70)
print("Analysis Complete")
print("="*70)
