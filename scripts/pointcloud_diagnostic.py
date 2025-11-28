#!/usr/bin/env python3
"""
Point Cloud Diagnostic Script
==============================
Analyzes LiDAR point cloud data to identify issues with:
1. Point cloud structure (ring/line distribution)
2. Time stamp format and ordering
3. Coordinate frame (X-forward or Y-forward)
4. Feature extraction potential
"""

import rosbag
import numpy as np
import struct
import sys

def analyze_pointcloud(bag_path, topic='/lidar_points', max_frames=10):
    print("="*60)
    print("Point Cloud Diagnostic")
    print("="*60)
    print(f"Bag: {bag_path}")
    print(f"Topic: {topic}")
    print()

    bag = rosbag.Bag(bag_path)

    frame_count = 0
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if frame_count >= max_frames:
            break

        print(f"\n{'='*60}")
        print(f"Frame {frame_count + 1}")
        print(f"{'='*60}")

        # Basic info
        print(f"Timestamp: {msg.header.stamp.to_sec()}")
        print(f"Frame ID: {msg.header.frame_id}")
        print(f"Width x Height: {msg.width} x {msg.height}")
        print(f"Point step: {msg.point_step} bytes")
        print(f"Is dense: {msg.is_dense}")

        # Field info
        print(f"\nFields:")
        field_info = {}
        for f in msg.fields:
            dtype_map = {1: 'int8', 2: 'uint8', 3: 'int16', 4: 'uint16',
                        5: 'int32', 6: 'uint32', 7: 'float32', 8: 'float64'}
            print(f"  {f.name}: offset={f.offset}, type={dtype_map.get(f.datatype, f.datatype)}")
            field_info[f.name] = (f.offset, f.datatype)

        # Parse points
        num_points = msg.width * msg.height
        data = np.frombuffer(msg.data, dtype=np.uint8)

        # Extract coordinates
        x_offset = field_info.get('x', (0, 7))[0]
        y_offset = field_info.get('y', (4, 7))[0]
        z_offset = field_info.get('z', (8, 7))[0]

        points = []
        lines = []
        timestamps = []

        for i in range(min(num_points, 10000)):
            base = i * msg.point_step
            x = struct.unpack('f', data[base+x_offset:base+x_offset+4])[0]
            y = struct.unpack('f', data[base+y_offset:base+y_offset+4])[0]
            z = struct.unpack('f', data[base+z_offset:base+z_offset+4])[0]

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z])

                # Extract line/ring
                if 'line' in field_info:
                    line_offset = field_info['line'][0]
                    line = struct.unpack('B', data[base+line_offset:base+line_offset+1])[0]
                    lines.append(line)
                elif 'ring' in field_info:
                    ring_offset = field_info['ring'][0]
                    ring = struct.unpack('H', data[base+ring_offset:base+ring_offset+2])[0]
                    lines.append(ring)

                # Extract timestamp
                if 'timestamp' in field_info:
                    ts_offset = field_info['timestamp'][0]
                    ts = struct.unpack('d', data[base+ts_offset:base+ts_offset+8])[0]
                    timestamps.append(ts)
                elif 'time' in field_info:
                    time_offset = field_info['time'][0]
                    t = struct.unpack('f', data[base+time_offset:base+time_offset+4])[0]
                    timestamps.append(t)

        points = np.array(points)

        print(f"\nPoint statistics (first 10000 points):")
        print(f"  Valid points: {len(points)}")

        if len(points) > 0:
            # Coordinate ranges
            print(f"\n  Coordinate ranges:")
            print(f"    X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}] m")
            print(f"    Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}] m")
            print(f"    Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}] m")

            # Distance distribution
            ranges = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
            print(f"\n  Range distribution:")
            print(f"    Min: {ranges.min():.2f} m")
            print(f"    Max: {ranges.max():.2f} m")
            print(f"    Mean: {ranges.mean():.2f} m")

            # Forward direction analysis
            print(f"\n  Forward direction analysis:")
            # Check which axis has most points in positive direction
            x_forward = np.sum(points[:,0] > 0) / len(points) * 100
            y_forward = np.sum(points[:,1] > 0) / len(points) * 100
            print(f"    Points with X > 0: {x_forward:.1f}%")
            print(f"    Points with Y > 0: {y_forward:.1f}%")

            # Determine likely forward axis
            if x_forward > 60 or x_forward < 40:
                fwd = "X" if x_forward > 50 else "-X"
            elif y_forward > 60 or y_forward < 40:
                fwd = "Y" if y_forward > 50 else "-Y"
            else:
                fwd = "Unclear"
            print(f"    Likely forward axis: {fwd}")

            # Line/ring distribution
            if len(lines) > 0:
                lines = np.array(lines)
                unique_lines = np.unique(lines)
                print(f"\n  Line/Ring distribution:")
                print(f"    Unique lines: {len(unique_lines)}")
                print(f"    Line values: {unique_lines[:20]}{'...' if len(unique_lines) > 20 else ''}")

                for line in unique_lines[:5]:
                    count = np.sum(lines == line)
                    print(f"    Line {line}: {count} points")

            # Timestamp analysis
            if len(timestamps) > 0:
                timestamps = np.array(timestamps)
                print(f"\n  Timestamp analysis:")
                print(f"    First: {timestamps[0]}")
                print(f"    Last: {timestamps[-1]}")

                if timestamps[0] > 1e9:  # Absolute timestamp (likely milliseconds)
                    print(f"    Format: Absolute (likely milliseconds)")
                    span = (timestamps[-1] - timestamps[0]) / 1000.0
                    print(f"    Scan duration: {span:.4f} seconds")
                else:
                    print(f"    Format: Relative (seconds)")
                    print(f"    Scan duration: {timestamps[-1] - timestamps[0]:.4f} seconds")

                # Check monotonicity
                diffs = np.diff(timestamps)
                if np.all(diffs >= 0):
                    print(f"    Timestamps: Monotonically increasing ✓")
                else:
                    negative_jumps = np.sum(diffs < 0)
                    print(f"    Timestamps: NOT monotonic! ({negative_jumps} backward jumps)")

            # Feature extraction potential
            print(f"\n  Feature extraction analysis:")

            # Compute local curvature (simplified)
            curvatures = []
            sorted_indices = np.argsort(np.arctan2(points[:,1], points[:,0]))
            sorted_points = points[sorted_indices]

            for i in range(5, len(sorted_points) - 5):
                neighbors = sorted_points[i-5:i+6]
                diff = neighbors - neighbors[5]
                curvature = np.sum(diff**2) / (10 * ranges[sorted_indices[i]]**2 + 1e-6)
                curvatures.append(curvature)

            curvatures = np.array(curvatures)
            edge_threshold = 0.1
            surf_threshold = 0.1

            edge_count = np.sum(curvatures > edge_threshold)
            surf_count = np.sum(curvatures < surf_threshold)

            print(f"    Edge features (curvature > {edge_threshold}): {edge_count}")
            print(f"    Surface features (curvature < {surf_threshold}): {surf_count}")

            if edge_count < 100:
                print(f"    ⚠️  WARNING: Very few edge features!")
            if surf_count < 1000:
                print(f"    ⚠️  WARNING: Very few surface features!")

        frame_count += 1

    bag.close()
    print(f"\n{'='*60}")
    print("Diagnostic Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 pointcloud_diagnostic.py <bag_file> [topic]")
        sys.exit(1)

    bag_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else '/lidar_points'

    analyze_pointcloud(bag_path, topic)
