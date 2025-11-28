#!/usr/bin/env python
import rosbag
import numpy as np
from collections import defaultdict

def detailed_timestamp_check(bag_path):
    """Perform detailed checks on timestamp matching"""
    print("="*70)
    print("DETAILED TIMESTAMP MATCHING ANALYSIS")
    print("="*70)

    bag = rosbag.Bag(bag_path)

    # Collect all timestamps with their header timestamps
    lidar_data = []
    imu_data = []

    print("\nExtracting header timestamps...")
    for topic, msg, t in bag.read_messages(topics=['/lidar_points', '/imu/data']):
        bag_time = t.to_sec()
        header_time = msg.header.stamp.to_sec()

        if topic == '/lidar_points':
            lidar_data.append({
                'bag_time': bag_time,
                'header_time': header_time,
                'diff': (bag_time - header_time) * 1000  # ms
            })
        else:  # /imu/data
            imu_data.append({
                'bag_time': bag_time,
                'header_time': header_time,
                'diff': (bag_time - header_time) * 1000  # ms
            })

    bag.close()

    # Analysis 1: Header vs Bag timestamp consistency
    print("\n1. HEADER vs BAG TIMESTAMP CONSISTENCY")
    print("-"*50)

    lidar_diffs = [d['diff'] for d in lidar_data]
    imu_diffs = [d['diff'] for d in imu_data]

    print("LiDAR (bag_time - header_time) in ms:")
    print(f"  Mean: {np.mean(lidar_diffs):.3f}")
    print(f"  Std:  {np.std(lidar_diffs):.3f}")
    print(f"  Min:  {np.min(lidar_diffs):.3f}")
    print(f"  Max:  {np.max(lidar_diffs):.3f}")

    print("\nIMU (bag_time - header_time) in ms:")
    print(f"  Mean: {np.mean(imu_diffs):.3f}")
    print(f"  Std:  {np.std(imu_diffs):.3f}")
    print(f"  Min:  {np.min(imu_diffs):.3f}")
    print(f"  Max:  {np.max(imu_diffs):.3f}")

    # Check if timestamps are monotonically increasing
    print("\n2. MONOTONICITY CHECK")
    print("-"*50)

    lidar_header_times = [d['header_time'] for d in lidar_data]
    imu_header_times = [d['header_time'] for d in imu_data]

    lidar_non_mono = 0
    imu_non_mono = 0

    for i in range(1, len(lidar_header_times)):
        if lidar_header_times[i] <= lidar_header_times[i-1]:
            lidar_non_mono += 1

    for i in range(1, len(imu_header_times)):
        if imu_header_times[i] <= imu_header_times[i-1]:
            imu_non_mono += 1

    print(f"Non-monotonic timestamps:")
    print(f"  LiDAR: {lidar_non_mono}/{len(lidar_header_times)-1}")
    print(f"  IMU:   {imu_non_mono}/{len(imu_header_times)-1}")

    # Analysis 3: Time offset pattern
    print("\n3. TIME OFFSET PATTERN ANALYSIS")
    print("-"*50)

    # Check if there's a consistent offset between LiDAR and IMU
    min_len = min(len(lidar_header_times), len(imu_header_times))

    # Sample every 100th point for offset analysis
    sample_indices = range(0, min_len, min_len // 100) if min_len > 100 else range(min_len)

    offsets = []
    for i in sample_indices:
        # Find closest IMU to this LiDAR time
        lidar_t = lidar_header_times[i // 20] if i // 20 < len(lidar_header_times) else lidar_header_times[-1]
        closest_idx = np.argmin([abs(imu_t - lidar_t) for imu_t in imu_header_times])
        offset = (imu_header_times[closest_idx] - lidar_t) * 1000  # ms
        offsets.append(offset)

    if offsets:
        print(f"Systematic offset (IMU - LiDAR) over time:")
        print(f"  Start of bag: {offsets[0]:.3f} ms")
        print(f"  Middle of bag: {offsets[len(offsets)//2]:.3f} ms")
        print(f"  End of bag: {offsets[-1]:.3f} ms")
        print(f"  Drift: {abs(offsets[-1] - offsets[0]):.3f} ms")

    # Analysis 4: Check for missing data periods
    print("\n4. DATA CONTINUITY CHECK")
    print("-"*50)

    # Check for gaps larger than expected
    lidar_gaps = []
    imu_gaps = []

    for i in range(1, len(lidar_header_times)):
        gap = (lidar_header_times[i] - lidar_header_times[i-1]) * 1000  # ms
        if gap > 150:  # Expected ~100ms, flag if > 150ms
            lidar_gaps.append((i, gap))

    for i in range(1, len(imu_header_times)):
        gap = (imu_header_times[i] - imu_header_times[i-1]) * 1000  # ms
        if gap > 15:  # Expected ~5ms, flag if > 15ms
            imu_gaps.append((i, gap))

    print(f"Data gaps detected:")
    print(f"  LiDAR (>150ms): {len(lidar_gaps)} gaps")
    if lidar_gaps and len(lidar_gaps) <= 5:
        for idx, gap in lidar_gaps[:5]:
            print(f"    At index {idx}: {gap:.1f} ms gap")

    print(f"  IMU (>15ms): {len(imu_gaps)} gaps")
    if imu_gaps and len(imu_gaps) <= 5:
        for idx, gap in imu_gaps[:5]:
            print(f"    At index {idx}: {gap:.1f} ms gap")

    # Analysis 5: Statistical synchronization quality
    print("\n5. SYNCHRONIZATION QUALITY METRICS")
    print("-"*50)

    # For each LiDAR point, find IMU messages within time window
    sync_stats = {
        'perfect': 0,  # Within 1ms
        'good': 0,     # Within 5ms
        'acceptable': 0,  # Within 10ms
        'poor': 0      # > 10ms
    }

    for lidar_t in lidar_header_times:
        min_diff = min([abs(imu_t - lidar_t) * 1000 for imu_t in imu_header_times])
        if min_diff <= 1:
            sync_stats['perfect'] += 1
        elif min_diff <= 5:
            sync_stats['good'] += 1
        elif min_diff <= 10:
            sync_stats['acceptable'] += 1
        else:
            sync_stats['poor'] += 1

    total = len(lidar_header_times)
    print("Synchronization quality distribution:")
    print(f"  Perfect (≤1ms):     {sync_stats['perfect']:5d} ({sync_stats['perfect']/total*100:5.1f}%)")
    print(f"  Good (1-5ms):       {sync_stats['good']:5d} ({sync_stats['good']/total*100:5.1f}%)")
    print(f"  Acceptable (5-10ms):{sync_stats['acceptable']:5d} ({sync_stats['acceptable']/total*100:5.1f}%)")
    print(f"  Poor (>10ms):       {sync_stats['poor']:5d} ({sync_stats['poor']/total*100:5.1f}%)")

    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)

    issues = []
    recommendations = []

    # Check for issues
    if abs(np.mean(lidar_diffs)) > 10 or abs(np.mean(imu_diffs)) > 10:
        issues.append("Large discrepancy between bag time and header time")
        recommendations.append("Check sensor driver timestamp settings")

    if lidar_non_mono > 0 or imu_non_mono > 0:
        issues.append("Non-monotonic timestamps detected")
        recommendations.append("Investigate timestamp generation in sensor drivers")

    if offsets and abs(offsets[-1] - offsets[0]) > 10:
        issues.append(f"Clock drift detected: {abs(offsets[-1] - offsets[0]):.1f} ms")
        recommendations.append("Check for clock synchronization issues (NTP/PTP)")

    if len(lidar_gaps) > 0 or len(imu_gaps) > 10:
        issues.append("Data continuity issues detected")
        recommendations.append("Check for sensor communication issues or CPU overload")

    if sync_stats['poor'] > total * 0.01:  # More than 1% poor sync
        issues.append("Poor time synchronization for some messages")
        recommendations.append("Consider implementing timestamp interpolation or filtering")

    if issues:
        print("\nISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✓ No major timestamp issues detected!")
        print("  - Timestamps are monotonic")
        print("  - Good synchronization between LiDAR and IMU")
        print("  - Consistent timing throughout the bag file")

    # Overall assessment
    sync_quality = (sync_stats['perfect'] + sync_stats['good']) / total * 100
    print(f"\nOVERALL SYNC QUALITY: {sync_quality:.1f}%")
    if sync_quality >= 95:
        print("  Status: EXCELLENT - Suitable for sensor fusion")
    elif sync_quality >= 90:
        print("  Status: GOOD - Minor improvements possible")
    elif sync_quality >= 80:
        print("  Status: ACCEPTABLE - Consider timestamp filtering")
    else:
        print("  Status: POOR - Requires timestamp correction")

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"
    detailed_timestamp_check(bag_path)