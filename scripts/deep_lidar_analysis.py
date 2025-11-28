#!/usr/bin/env python3
"""
Deep LiDAR Frame Analysis
Determines the actual mounting orientation and coordinate frame
"""

import rosbag
import numpy as np
import struct
import sys

def deep_analysis(bag_path, topic='/lidar_points', num_frames=3):
    print("=" * 70)
    print("Deep LiDAR Coordinate Frame Analysis")
    print("=" * 70)

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

        # Collect ALL valid points (no distance filtering)
        all_x, all_y, all_z = [], [], []
        ranges = []

        for i in range(num_points):
            base = i * msg.point_step
            x = struct.unpack('f', data[base+x_offset:base+x_offset+4])[0]
            y = struct.unpack('f', data[base+y_offset:base+y_offset+4])[0]
            z = struct.unpack('f', data[base+z_offset:base+z_offset+4])[0]

            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                r = np.sqrt(x*x + y*y + z*z)
                if r > 0.1:  # Minimal filtering
                    all_x.append(x)
                    all_y.append(y)
                    all_z.append(z)
                    ranges.append(r)

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        all_z = np.array(all_z)
        ranges = np.array(ranges)

        print(f"\nTotal valid points: {len(all_x)}")

        # Full statistics
        print(f"\n--- Full Point Statistics ---")
        print(f"X: min={all_x.min():.2f}, max={all_x.max():.2f}, mean={all_x.mean():.2f}, std={all_x.std():.2f}")
        print(f"Y: min={all_y.min():.2f}, max={all_y.max():.2f}, mean={all_y.mean():.2f}, std={all_y.std():.2f}")
        print(f"Z: min={all_z.min():.2f}, max={all_z.max():.2f}, mean={all_z.mean():.2f}, std={all_z.std():.2f}")
        print(f"Range: min={ranges.min():.2f}, max={ranges.max():.2f}, mean={ranges.mean():.2f}")

        # Distribution by quadrant
        print(f"\n--- Point Distribution ---")
        total = len(all_x)

        # X axis analysis
        x_pos = np.sum(all_x > 0)
        x_neg = np.sum(all_x < 0)
        print(f"X > 0: {x_pos} ({x_pos/total*100:.1f}%)")
        print(f"X < 0: {x_neg} ({x_neg/total*100:.1f}%)")

        # Y axis analysis
        y_pos = np.sum(all_y > 0)
        y_neg = np.sum(all_y < 0)
        print(f"Y > 0: {y_pos} ({y_pos/total*100:.1f}%)")
        print(f"Y < 0: {y_neg} ({y_neg/total*100:.1f}%)")

        # Z axis analysis
        z_pos = np.sum(all_z > 0)
        z_neg = np.sum(all_z < 0)
        print(f"Z > 0: {z_pos} ({z_pos/total*100:.1f}%)")
        print(f"Z < 0: {z_neg} ({z_neg/total*100:.1f}%)")

        # Far points analysis (> 5m)
        far_mask = ranges > 5
        if np.sum(far_mask) > 100:
            print(f"\n--- Far Points (> 5m) Analysis ---")
            far_x = all_x[far_mask]
            far_y = all_y[far_mask]
            far_z = all_z[far_mask]
            far_count = len(far_x)

            print(f"Far points count: {far_count}")
            print(f"X: min={far_x.min():.2f}, max={far_x.max():.2f}, mean={far_x.mean():.2f}")
            print(f"Y: min={far_y.min():.2f}, max={far_y.max():.2f}, mean={far_y.mean():.2f}")
            print(f"Z: min={far_z.min():.2f}, max={far_z.max():.2f}, mean={far_z.mean():.2f}")

            far_x_pos = np.sum(far_x > 0) / far_count * 100
            far_x_neg = np.sum(far_x < 0) / far_count * 100
            far_y_pos = np.sum(far_y > 0) / far_count * 100
            far_y_neg = np.sum(far_y < 0) / far_count * 100

            print(f"Far X > 0: {far_x_pos:.1f}%, Far X < 0: {far_x_neg:.1f}%")
            print(f"Far Y > 0: {far_y_pos:.1f}%, Far Y < 0: {far_y_neg:.1f}%")

        # Ground plane analysis
        ground_mask = np.abs(all_z) < 0.3
        if np.sum(ground_mask) > 100:
            print(f"\n--- Ground Plane (|Z| < 0.3m) ---")
            ground_x = all_x[ground_mask]
            ground_y = all_y[ground_mask]
            ground_count = len(ground_x)
            print(f"Ground points: {ground_count}")

            gnd_x_pos = np.sum(ground_x > 0) / ground_count * 100
            gnd_x_neg = np.sum(ground_x < 0) / ground_count * 100
            print(f"Ground X > 0: {gnd_x_pos:.1f}%, Ground X < 0: {gnd_x_neg:.1f}%")

        # Determine forward direction
        print(f"\n--- Forward Direction Analysis ---")

        # Check which direction has more far points
        if np.sum(far_mask) > 100:
            mean_far_x = far_x.mean()
            mean_far_y = far_y.mean()

            if abs(mean_far_x) > abs(mean_far_y):
                if mean_far_x > 0:
                    print(f"CONCLUSION: LiDAR +X points FORWARD (mean far X = {mean_far_x:.2f})")
                    forward = "+X"
                else:
                    print(f"CONCLUSION: LiDAR -X points FORWARD (mean far X = {mean_far_x:.2f})")
                    forward = "-X"
            else:
                if mean_far_y > 0:
                    print(f"CONCLUSION: LiDAR +Y points FORWARD (mean far Y = {mean_far_y:.2f})")
                    forward = "+Y"
                else:
                    print(f"CONCLUSION: LiDAR -Y points FORWARD (mean far Y = {mean_far_y:.2f})")
                    forward = "-Y"
        else:
            forward = "unknown"
            print("Not enough far points to determine forward direction")

        frame_count += 1

    bag.close()

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if forward == "-X":
        print("""
LiDAR -X points forward. Options:
1. Rotate LiDAR points 180° around Z: x'=-x, y'=-y, z'=z
2. Use extrinsicRot to rotate IMU to match LiDAR frame

Current issue: The 180° rotation might be WRONG if the LiDAR
is actually scanning in +X direction but mounted pointing backward.

Check your physical LiDAR mounting!
""")
    elif forward == "+X":
        print("""
LiDAR +X already points forward. No rotation needed.
Set extrinsicRot and extrinsicRPY to identity matrices.
""")
    elif forward == "+Y" or forward == "-Y":
        print(f"""
LiDAR {forward} points forward. Need 90° rotation around Z.
This is unusual - please verify LiDAR mounting.
""")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 deep_lidar_analysis.py <bag_file>")
        sys.exit(1)

    deep_analysis(sys.argv[1])
