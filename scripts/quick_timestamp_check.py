#!/usr/bin/env python
import rosbag
import numpy as np

def quick_timestamp_check(bag_path):
    """Quick timestamp matching check with sampling"""
    print("="*70)
    print("QUICK TIMESTAMP MATCHING ANALYSIS (SAMPLED)")
    print("="*70)

    bag = rosbag.Bag(bag_path)

    # Sample data for faster processing
    lidar_data = []
    imu_data = []
    sample_rate = 10  # Sample every 10th message for IMU

    print("\nExtracting sampled header timestamps...")
    msg_count = 0
    for topic, msg, t in bag.read_messages(topics=['/lidar_points', '/imu/data']):
        bag_time = t.to_sec()
        header_time = msg.header.stamp.to_sec()

        if topic == '/lidar_points':
            # Keep all LiDAR messages (only ~8800)
            lidar_data.append({
                'bag_time': bag_time,
                'header_time': header_time,
                'diff': (bag_time - header_time) * 1000  # ms
            })
        else:  # /imu/data
            # Sample IMU messages
            if msg_count % sample_rate == 0:
                imu_data.append({
                    'bag_time': bag_time,
                    'header_time': header_time,
                    'diff': (bag_time - header_time) * 1000  # ms
                })
            msg_count += 1

    bag.close()
    print(f"Sampled {len(lidar_data)} LiDAR and {len(imu_data)} IMU messages")

    # Quick Analysis
    print("\n1. HEADER vs BAG TIME CONSISTENCY")
    print("-"*50)

    lidar_diffs = [d['diff'] for d in lidar_data]
    imu_diffs = [d['diff'] for d in imu_data]

    print("LiDAR (bag_time - header_time) in ms:")
    print(f"  Mean: {np.mean(lidar_diffs):.3f}")
    print(f"  Std:  {np.std(lidar_diffs):.3f}")
    print(f"  Range: [{np.min(lidar_diffs):.3f}, {np.max(lidar_diffs):.3f}]")

    print("\nIMU (bag_time - header_time) in ms:")
    print(f"  Mean: {np.mean(imu_diffs):.3f}")
    print(f"  Std:  {np.std(imu_diffs):.3f}")
    print(f"  Range: [{np.min(imu_diffs):.3f}, {np.max(imu_diffs):.3f}]")

    # Check monotonicity
    print("\n2. TIMESTAMP MONOTONICITY")
    print("-"*50)

    lidar_times = [d['header_time'] for d in lidar_data]
    imu_times = [d['header_time'] for d in imu_data]

    lidar_mono = all(lidar_times[i] > lidar_times[i-1] for i in range(1, len(lidar_times)))
    imu_mono = all(imu_times[i] > imu_times[i-1] for i in range(1, len(imu_times)))

    print(f"LiDAR timestamps monotonic: {lidar_mono}")
    print(f"IMU timestamps monotonic: {imu_mono}")

    # Time offset check
    print("\n3. TIME OFFSET PATTERN")
    print("-"*50)

    # Check offset at different points
    points = [0, len(lidar_times)//4, len(lidar_times)//2, 3*len(lidar_times)//4, -1]
    print("Offset between LiDAR and nearest IMU at different times:")

    for i, idx in enumerate(points):
        lidar_t = lidar_times[idx]
        closest_idx = np.argmin([abs(imu_t - lidar_t) for imu_t in imu_times])
        offset = (imu_times[closest_idx] - lidar_t) * 1000  # ms
        position = ["Start", "25%", "50%", "75%", "End"][i]
        print(f"  {position:5s}: {offset:7.3f} ms")

    # Quick sync quality check
    print("\n4. SYNCHRONIZATION QUALITY")
    print("-"*50)

    sync_quality = {'<1ms': 0, '1-5ms': 0, '5-10ms': 0, '>10ms': 0}

    for lidar_t in lidar_times[::10]:  # Sample every 10th LiDAR
        min_diff = min([abs(imu_t - lidar_t) * 1000 for imu_t in imu_times])
        if min_diff < 1:
            sync_quality['<1ms'] += 1
        elif min_diff < 5:
            sync_quality['1-5ms'] += 1
        elif min_diff < 10:
            sync_quality['5-10ms'] += 1
        else:
            sync_quality['>10ms'] += 1

    total = sum(sync_quality.values())
    print("Time difference distribution (sampled):")
    for key, val in sync_quality.items():
        print(f"  {key:6s}: {val:4d} ({val/total*100:5.1f}%)")

    # Summary
    print("\n" + "="*70)
    print("QUICK ASSESSMENT")
    print("="*70)

    issues = []

    if not lidar_mono or not imu_mono:
        issues.append("Non-monotonic timestamps detected")

    if abs(np.mean(lidar_diffs)) > 10 or abs(np.mean(imu_diffs)) > 10:
        issues.append("Large bag/header time discrepancy")

    good_sync = (sync_quality['<1ms'] + sync_quality['1-5ms']) / total * 100

    if good_sync < 90:
        issues.append(f"Synchronization quality below 90% ({good_sync:.1f}%)")

    if issues:
        print("POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ No major issues detected in sampled data")

    print(f"\nOVERALL SYNC QUALITY (sampled): {good_sync:.1f}%")

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"
    quick_timestamp_check(bag_path)