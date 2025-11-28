#!/usr/bin/env python3
import rosbag
import numpy as np
from sensor_msgs.msg import PointCloud2
import struct

def analyze_pointcloud(bag_path, topic='/lidar_points', max_messages=10):
    """Analyze point cloud data to verify LiDAR parameters."""

    bag = rosbag.Bag(bag_path)

    # Statistics to collect
    points_per_scan = []
    line_counts = {}
    point_ranges = []
    timestamps = []

    print(f"Analyzing {topic} from {bag_path}")
    print("="*60)

    msg_count = 0
    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if msg_count >= max_messages:
            break

        msg_count += 1
        timestamps.append(t.to_sec())

        # Get basic info
        width = msg.width
        height = msg.height
        point_step = msg.point_step
        row_step = msg.row_step

        # Parse fields
        fields_info = {}
        for field in msg.fields:
            fields_info[field.name] = {
                'offset': field.offset,
                'datatype': field.datatype,
                'count': field.count
            }

        # Count points
        num_points = width * height
        points_per_scan.append(num_points)

        # Analyze first message in detail
        if msg_count == 1:
            print(f"Message #{msg_count} Details:")
            print(f"  Width (points): {width}")
            print(f"  Height: {height}")
            print(f"  Total points: {num_points}")
            print(f"  Point step: {point_step} bytes")
            print(f"  Row step: {row_step} bytes")
            print(f"  Is dense: {msg.is_dense}")
            print(f"  Frame ID: {msg.header.frame_id}")
            print(f"\n  Fields:")
            for fname, finfo in fields_info.items():
                print(f"    {fname}: offset={finfo['offset']}, type={finfo['datatype']}, count={finfo['count']}")

        # Parse actual point data for line analysis
        if 'line' in fields_info:
            line_offset = fields_info['line']['offset']
            lines_in_scan = {}

            # Sample points to analyze line distribution
            for i in range(0, len(msg.data), point_step):
                if i + line_offset < len(msg.data):
                    line_id = msg.data[i + line_offset]
                    lines_in_scan[line_id] = lines_in_scan.get(line_id, 0) + 1

            if msg_count <= 3:  # Show details for first 3 messages
                print(f"\n  Line distribution (Message #{msg_count}):")
                for line_id in sorted(lines_in_scan.keys()):
                    count = lines_in_scan[line_id]
                    percentage = (count / num_points) * 100
                    print(f"    Line {line_id:2d}: {count:5d} points ({percentage:5.1f}%)")

            # Update overall line statistics
            for line_id, count in lines_in_scan.items():
                if line_id not in line_counts:
                    line_counts[line_id] = []
                line_counts[line_id].append(count)

        # Parse XYZ for range analysis (sample first 1000 points)
        if 'x' in fields_info and 'y' in fields_info and 'z' in fields_info:
            x_offset = fields_info['x']['offset']
            y_offset = fields_info['y']['offset']
            z_offset = fields_info['z']['offset']

            ranges = []
            for i in range(0, min(1000*point_step, len(msg.data)), point_step):
                x = struct.unpack('f', msg.data[i+x_offset:i+x_offset+4])[0]
                y = struct.unpack('f', msg.data[i+y_offset:i+y_offset+4])[0]
                z = struct.unpack('f', msg.data[i+z_offset:i+z_offset+4])[0]
                r = np.sqrt(x*x + y*y + z*z)
                if r > 0.01:  # Filter out zero/invalid points
                    ranges.append(r)

            if ranges:
                point_ranges.extend(ranges)
                if msg_count == 1:
                    print(f"\n  Range statistics (sampled):")
                    print(f"    Min range: {min(ranges):.2f} m")
                    print(f"    Max range: {max(ranges):.2f} m")
                    print(f"    Mean range: {np.mean(ranges):.2f} m")

    bag.close()

    # Calculate overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS:")
    print("="*60)

    # Points per scan
    print(f"\nPoints per scan:")
    print(f"  Mean: {np.mean(points_per_scan):.0f}")
    print(f"  Min: {min(points_per_scan)}")
    print(f"  Max: {max(points_per_scan)}")
    print(f"  Std: {np.std(points_per_scan):.1f}")

    # Line statistics
    if line_counts:
        print(f"\nLine statistics:")
        print(f"  Number of unique lines: {len(line_counts)}")
        print(f"  Line IDs: {sorted(line_counts.keys())}")

        # Average points per line
        print(f"\n  Points per line (averaged across {msg_count} messages):")
        total_points_avg = np.mean(points_per_scan)
        for line_id in sorted(line_counts.keys()):
            avg_points = np.mean(line_counts[line_id])
            percentage = (avg_points / total_points_avg) * 100
            print(f"    Line {line_id:2d}: {avg_points:6.0f} points ({percentage:5.1f}%)")

        # Horizon resolution estimation
        avg_points_per_line = total_points_avg / len(line_counts)
        print(f"\n  Estimated horizontal resolution: {avg_points_per_line:.0f} points/line")

    # Range statistics
    if point_ranges:
        print(f"\nRange statistics (all sampled points):")
        print(f"  Min: {min(point_ranges):.2f} m")
        print(f"  Max: {max(point_ranges):.2f} m")
        print(f"  Mean: {np.mean(point_ranges):.2f} m")
        print(f"  Percentiles:")
        print(f"    1%: {np.percentile(point_ranges, 1):.2f} m")
        print(f"    99%: {np.percentile(point_ranges, 99):.2f} m")

    # Frequency calculation
    if len(timestamps) > 1:
        time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_freq = 1.0 / np.mean(time_diffs)
        print(f"\nScan frequency:")
        print(f"  Average: {avg_freq:.2f} Hz")
        print(f"  Expected interval: {1.0/avg_freq*1000:.1f} ms")

    return {
        'points_per_scan': points_per_scan,
        'line_counts': line_counts,
        'point_ranges': point_ranges,
        'frequency': avg_freq if len(timestamps) > 1 else None
    }

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"
    stats = analyze_pointcloud(bag_path, topic='/lidar_points', max_messages=10)

    # Verify against params.yaml configuration
    print("\n" + "="*60)
    print("PARAMETER VERIFICATION:")
    print("="*60)

    config_n_scan = 16
    config_horizon_scan = 5000
    config_min_range = 1.0
    config_max_range = 1000.0

    actual_lines = len(stats['line_counts']) if stats['line_counts'] else 0
    actual_horizon = np.mean(stats['points_per_scan']) / actual_lines if actual_lines > 0 else 0

    print(f"\n1. N_SCAN (Number of scan lines):")
    print(f"   Config: {config_n_scan}")
    print(f"   Actual: {actual_lines}")
    print(f"   Status: {'✓ MATCH' if actual_lines == config_n_scan else '✗ MISMATCH'}")

    print(f"\n2. Horizon_SCAN (Points per line):")
    print(f"   Config: {config_horizon_scan}")
    print(f"   Actual: {actual_horizon:.0f}")
    print(f"   Status: {'✓ CLOSE' if abs(actual_horizon - config_horizon_scan) < 100 else '⚠ DIFFERENCE'}")

    if stats['point_ranges']:
        actual_min = np.percentile(stats['point_ranges'], 1)
        actual_max = np.percentile(stats['point_ranges'], 99)

        print(f"\n3. Range limits:")
        print(f"   Min range - Config: {config_min_range:.1f} m")
        print(f"   Min range - Actual (1%): {actual_min:.2f} m")
        print(f"   Max range - Config: {config_max_range:.1f} m")
        print(f"   Max range - Actual (99%): {actual_max:.2f} m")
        print(f"   Status: {'✓ REASONABLE' if actual_min >= 0.5 and actual_max <= config_max_range else '⚠ CHECK NEEDED'}")

    if stats['frequency']:
        print(f"\n4. Scan frequency:")
        print(f"   Actual: {stats['frequency']:.2f} Hz")
        print(f"   Status: {'✓ NORMAL' if 9 <= stats['frequency'] <= 11 else '⚠ UNUSUAL'}")