#!/usr/bin/env python3
"""
Check Range Image Density for Livox
Analyzes how sparse the range image is after angle-based projection
"""

import rosbag
import numpy as np
import struct
import sys

def analyze_range_image(bag_path, topic='/lidar_points', num_frames=3):
    print("=" * 70)
    print("Range Image Density Analysis for Livox")
    print("=" * 70)

    # Parameters from params.yaml
    N_SCAN = 16
    Horizon_SCAN = 5000

    bag = rosbag.Bag(bag_path)

    frame_count = 0
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if frame_count >= num_frames:
            break

        print(f"\n{'='*70}")
        print(f"Frame {frame_count + 1}")
        print(f"{'='*70}")

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

        print(f"Total points in message: {num_points}")

        # Simulate range image projection
        range_mat = np.full((N_SCAN, Horizon_SCAN), np.inf)
        ang_res_x = 360.0 / Horizon_SCAN

        valid_points = 0
        rejected_range = 0
        rejected_ring = 0
        rejected_duplicate = 0

        column_counts = np.zeros((N_SCAN, Horizon_SCAN), dtype=int)

        for i in range(num_points):
            base = i * msg.point_step
            x = struct.unpack('f', data[base+x_offset:base+x_offset+4])[0]
            y = struct.unpack('f', data[base+y_offset:base+y_offset+4])[0]
            z = struct.unpack('f', data[base+z_offset:base+z_offset+4])[0]
            ring = data[base+line_offset]

            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue

            # Rotate coordinates (as in code)
            x_rot = -x
            y_rot = -y

            # Calculate range
            r = np.sqrt(x_rot**2 + y_rot**2 + z**2)
            if r < 0.3 or r > 100:
                rejected_range += 1
                continue

            if ring < 0 or ring >= N_SCAN:
                rejected_ring += 1
                continue

            # Calculate column index (same as code)
            horizon_angle = np.arctan2(x_rot, y_rot) * 180 / np.pi
            column_idn = int(-round((horizon_angle - 90.0) / ang_res_x) + Horizon_SCAN / 2)
            if column_idn >= Horizon_SCAN:
                column_idn -= Horizon_SCAN
            if column_idn < 0:
                column_idn += Horizon_SCAN

            if column_idn < 0 or column_idn >= Horizon_SCAN:
                continue

            column_counts[ring, column_idn] += 1

            if range_mat[ring, column_idn] != np.inf:
                rejected_duplicate += 1
                continue

            range_mat[ring, column_idn] = r
            valid_points += 1

        print(f"\nProjection Results:")
        print(f"  Valid points in range image: {valid_points}")
        print(f"  Rejected (range): {rejected_range}")
        print(f"  Rejected (ring): {rejected_ring}")
        print(f"  Rejected (duplicate): {rejected_duplicate}")

        # Analyze range image density
        total_cells = N_SCAN * Horizon_SCAN
        filled_cells = np.sum(range_mat != np.inf)
        density = filled_cells / total_cells * 100

        print(f"\nRange Image Analysis:")
        print(f"  Total cells: {total_cells}")
        print(f"  Filled cells: {filled_cells}")
        print(f"  Density: {density:.2f}%")

        # Analyze per-ring
        print(f"\nPer-Ring Analysis:")
        for ring in range(N_SCAN):
            ring_filled = np.sum(range_mat[ring, :] != np.inf)
            ring_density = ring_filled / Horizon_SCAN * 100
            max_duplicates = np.max(column_counts[ring, :])
            avg_duplicates = np.mean(column_counts[ring, column_counts[ring, :] > 0]) if np.any(column_counts[ring, :] > 0) else 0
            print(f"  Ring {ring:2d}: {ring_filled:5d} points ({ring_density:5.2f}%), max_dup={max_duplicates}, avg_dup={avg_duplicates:.1f}")

        # Analyze gaps
        print(f"\nGap Analysis (consecutive empty cells):")
        for ring in range(N_SCAN):
            valid_cols = np.where(range_mat[ring, :] != np.inf)[0]
            if len(valid_cols) > 1:
                gaps = np.diff(valid_cols)
                max_gap = np.max(gaps)
                avg_gap = np.mean(gaps)
                print(f"  Ring {ring:2d}: max_gap={max_gap:4d}, avg_gap={avg_gap:.1f}")

        frame_count += 1

    bag.close()

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
If density is very low (<1%) and gaps are large (>100):
  -> Range image is too sparse for LIO-SAM's feature extraction
  -> Smoothness calculation will be unreliable
  -> Consider using a smaller Horizon_SCAN or different approach
""")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 check_range_image.py <bag_file>")
        sys.exit(1)

    analyze_range_image(sys.argv[1])
