#!/usr/bin/env python3
"""
Analyze Livox point ordering within each scan line
Check if adjacent points are spatially adjacent
"""

import rosbag
import numpy as np
import struct
import sys

def analyze_point_ordering(bag_path, topic='/lidar_points', num_frames=1):
    print("=" * 70)
    print("Livox Point Ordering Analysis")
    print("=" * 70)

    bag = rosbag.Bag(bag_path)

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        # Parse fields
        field_info = {}
        for f in msg.fields:
            field_info[f.name] = (f.offset, f.datatype)

        data = np.frombuffer(msg.data, dtype=np.uint8)
        num_points = msg.width * msg.height

        x_offset = field_info.get('x', (0, 7))[0]
        y_offset = field_info.get('y', (4, 7))[0]
        z_offset = field_info.get('z', (8, 7))[0]
        line_offset = field_info.get('line', (17, 1))[0]
        ts_offset = field_info.get('timestamp', (18, 8))[0]

        # Collect points per line
        points_per_line = {i: [] for i in range(16)}

        for i in range(num_points):
            base = i * msg.point_step
            x = struct.unpack('f', data[base+x_offset:base+x_offset+4])[0]
            y = struct.unpack('f', data[base+y_offset:base+y_offset+4])[0]
            z = struct.unpack('f', data[base+z_offset:base+z_offset+4])[0]
            line = data[base+line_offset]
            ts = struct.unpack('d', data[base+ts_offset:base+ts_offset+8])[0]

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Calculate horizontal angle
            angle = np.arctan2(y, x) * 180 / np.pi

            if line < 16:
                points_per_line[line].append({
                    'x': x, 'y': y, 'z': z,
                    'angle': angle,
                    'timestamp': ts,
                    'idx': i
                })

        print(f"\nTotal points: {num_points}")

        # Analyze each line
        for line in range(16):
            pts = points_per_line[line]
            if len(pts) < 10:
                continue

            print(f"\n--- Line {line} ({len(pts)} points) ---")

            # Check if points are sorted by angle
            angles = [p['angle'] for p in pts]
            timestamps = [p['timestamp'] for p in pts]

            # Calculate angle differences between adjacent points
            angle_diffs = np.diff(angles)

            # Count backward jumps (angle decreases)
            backward_jumps = np.sum(np.abs(angle_diffs) > 10)  # More than 10° jump

            print(f"  Angle range: [{min(angles):.1f}, {max(angles):.1f}]°")
            print(f"  Large angle jumps (>10°): {backward_jumps}")

            # Check first 20 points
            print(f"  First 10 angles: ", end="")
            for i in range(min(10, len(angles))):
                print(f"{angles[i]:.1f}° ", end="")
            print()

            # Check spatial adjacency
            spatial_jumps = 0
            for i in range(1, min(100, len(pts))):
                dx = pts[i]['x'] - pts[i-1]['x']
                dy = pts[i]['y'] - pts[i-1]['y']
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1.0:  # More than 1m jump
                    spatial_jumps += 1

            print(f"  Large spatial jumps (>1m) in first 100: {spatial_jumps}")

        break  # Only analyze first frame

    bag.close()

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
If there are many large angle/spatial jumps:
  -> Adjacent points in storage are NOT spatially adjacent
  -> Need to sort points by angle within each line before processing
""")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 check_point_ordering.py <bag_file>")
        sys.exit(1)

    analyze_point_ordering(sys.argv[1])
