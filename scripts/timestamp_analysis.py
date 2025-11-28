#!/usr/bin/env python
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

def extract_timestamps(bag_path):
    """Extract timestamps from the bag file for specified topics"""
    timestamps = defaultdict(list)

    print(f"Reading bag file: {bag_path}")
    bag = rosbag.Bag(bag_path)

    # Get total message count for progress tracking
    total_msgs = bag.get_message_count(topic_filters=['/lidar_points', '/imu/data'])
    processed = 0

    for topic, msg, t in bag.read_messages(topics=['/lidar_points', '/imu/data']):
        timestamps[topic].append(t.to_sec())
        processed += 1
        if processed % 10000 == 0:
            print(f"Processed {processed}/{total_msgs} messages...")

    bag.close()
    print(f"Extraction complete. Processed {processed} messages.")

    return timestamps

def analyze_timestamps(timestamps):
    """Analyze timestamp alignment and frequency"""
    lidar_ts = np.array(timestamps['/lidar_points'])
    imu_ts = np.array(timestamps['/imu/data'])

    print("\n" + "="*60)
    print("TIMESTAMP ANALYSIS REPORT")
    print("="*60)

    # Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-"*40)
    print(f"LiDAR messages: {len(lidar_ts)}")
    print(f"IMU messages: {len(imu_ts)}")
    print(f"IMU/LiDAR ratio: {len(imu_ts)/len(lidar_ts):.2f}")

    # Time range
    print(f"\nTime range:")
    print(f"  LiDAR: {lidar_ts[0]:.6f} to {lidar_ts[-1]:.6f}")
    print(f"  IMU:   {imu_ts[0]:.6f} to {imu_ts[-1]:.6f}")
    print(f"  Start difference: {abs(lidar_ts[0] - imu_ts[0])*1000:.3f} ms")
    print(f"  End difference: {abs(lidar_ts[-1] - imu_ts[-1])*1000:.3f} ms")

    # Frequency analysis
    print("\n2. FREQUENCY ANALYSIS")
    print("-"*40)
    lidar_freq = 1.0 / np.mean(np.diff(lidar_ts))
    imu_freq = 1.0 / np.mean(np.diff(imu_ts))
    lidar_freq_std = np.std(1.0 / np.diff(lidar_ts))
    imu_freq_std = np.std(1.0 / np.diff(imu_ts))

    print(f"Average frequency:")
    print(f"  LiDAR: {lidar_freq:.2f} Hz (±{lidar_freq_std:.2f} Hz)")
    print(f"  IMU:   {imu_freq:.2f} Hz (±{imu_freq_std:.2f} Hz)")

    # Time interval analysis
    lidar_intervals = np.diff(lidar_ts) * 1000  # Convert to ms
    imu_intervals = np.diff(imu_ts) * 1000

    print(f"\nTime intervals (ms):")
    print(f"  LiDAR: mean={np.mean(lidar_intervals):.2f}, std={np.std(lidar_intervals):.2f}")
    print(f"         min={np.min(lidar_intervals):.2f}, max={np.max(lidar_intervals):.2f}")
    print(f"  IMU:   mean={np.mean(imu_intervals):.2f}, std={np.std(imu_intervals):.2f}")
    print(f"         min={np.min(imu_intervals):.2f}, max={np.max(imu_intervals):.2f}")

    # Time synchronization analysis
    print("\n3. TIME SYNCHRONIZATION")
    print("-"*40)

    # For each lidar message, find the closest IMU message
    time_diffs = []
    for lidar_t in lidar_ts:
        idx = np.argmin(np.abs(imu_ts - lidar_t))
        time_diff = (imu_ts[idx] - lidar_t) * 1000  # Convert to ms
        time_diffs.append(time_diff)

    time_diffs = np.array(time_diffs)

    print(f"Time difference between LiDAR and nearest IMU (ms):")
    print(f"  Mean: {np.mean(time_diffs):.3f}")
    print(f"  Std:  {np.std(time_diffs):.3f}")
    print(f"  Min:  {np.min(time_diffs):.3f}")
    print(f"  Max:  {np.max(time_diffs):.3f}")
    print(f"  |Mean|: {np.mean(np.abs(time_diffs)):.3f}")

    # Check how many lidar messages have IMU within certain thresholds
    thresholds = [1, 5, 10, 20, 50]  # ms
    print(f"\nLiDAR messages with IMU within threshold:")
    for thresh in thresholds:
        count = np.sum(np.abs(time_diffs) <= thresh)
        percentage = (count / len(time_diffs)) * 100
        print(f"  ±{thresh:2d} ms: {count}/{len(time_diffs)} ({percentage:.1f}%)")

    # Check for time jumps or discontinuities
    print("\n4. CONTINUITY CHECK")
    print("-"*40)

    # Check for large time jumps
    lidar_jumps = lidar_intervals[lidar_intervals > 200]  # > 200ms
    imu_jumps = imu_intervals[imu_intervals > 10]  # > 10ms

    print(f"Large time jumps detected:")
    print(f"  LiDAR (>200ms): {len(lidar_jumps)} jumps")
    if len(lidar_jumps) > 0:
        print(f"    Max jump: {np.max(lidar_jumps):.2f} ms")
    print(f"  IMU (>10ms): {len(imu_jumps)} jumps")
    if len(imu_jumps) > 0:
        print(f"    Max jump: {np.max(imu_jumps):.2f} ms")

    # Check for backwards time
    lidar_backwards = np.sum(lidar_intervals < 0)
    imu_backwards = np.sum(imu_intervals < 0)
    print(f"\nBackwards timestamps:")
    print(f"  LiDAR: {lidar_backwards}")
    print(f"  IMU:   {imu_backwards}")

    return {
        'lidar_ts': lidar_ts,
        'imu_ts': imu_ts,
        'time_diffs': time_diffs,
        'lidar_intervals': lidar_intervals,
        'imu_intervals': imu_intervals
    }

def create_plots(data):
    """Create visualization plots"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Timestamp Analysis: /lidar_points vs /imu/data', fontsize=16)

    # Plot 1: Timestamp progression
    ax = axes[0, 0]
    t0 = min(data['lidar_ts'][0], data['imu_ts'][0])
    ax.plot(data['lidar_ts'] - t0, range(len(data['lidar_ts'])), 'b-', label='LiDAR', alpha=0.7)
    ax.plot(data['imu_ts'] - t0, range(len(data['imu_ts'])), 'r-', label='IMU', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Message Index')
    ax.set_title('Timestamp Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Time difference histogram
    ax = axes[0, 1]
    ax.hist(data['time_diffs'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label='Perfect sync')
    ax.set_xlabel('Time Difference (ms)')
    ax.set_ylabel('Count')
    ax.set_title('LiDAR to Nearest IMU Time Difference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: LiDAR interval distribution
    ax = axes[1, 0]
    ax.hist(data['lidar_intervals'], bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(x=np.mean(data['lidar_intervals']), color='r', linestyle='--',
               label=f"Mean: {np.mean(data['lidar_intervals']):.2f} ms")
    ax.set_xlabel('Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('LiDAR Message Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: IMU interval distribution
    ax = axes[1, 1]
    ax.hist(data['imu_intervals'], bins=50, edgecolor='black', alpha=0.7, color='red')
    ax.axvline(x=np.mean(data['imu_intervals']), color='b', linestyle='--',
               label=f"Mean: {np.mean(data['imu_intervals']):.2f} ms")
    ax.set_xlabel('Interval (ms)')
    ax.set_ylabel('Count')
    ax.set_title('IMU Message Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Time difference over time
    ax = axes[2, 0]
    ax.plot(data['lidar_ts'] - t0, data['time_diffs'], 'g-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', label='Perfect sync')
    ax.fill_between(data['lidar_ts'] - t0, -5, 5, alpha=0.2, color='green', label='±5ms zone')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time Difference (ms)')
    ax.set_title('Time Synchronization Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Cumulative distribution
    ax = axes[2, 1]
    sorted_diffs = np.sort(np.abs(data['time_diffs']))
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100
    ax.plot(sorted_diffs, cumulative, 'b-', linewidth=2)
    ax.axvline(x=5, color='r', linestyle='--', alpha=0.5, label='5ms')
    ax.axvline(x=10, color='r', linestyle='--', alpha=0.5, label='10ms')
    ax.set_xlabel('Absolute Time Difference (ms)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Distribution of Time Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/autodl-tmp/catkin_ws/src/LIO-SAM/output/timestamp_analysis.png', dpi=150)
    print("\nPlot saved to: /root/autodl-tmp/catkin_ws/src/LIO-SAM/output/timestamp_analysis.png")
    plt.show()

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"

    # Extract timestamps
    timestamps = extract_timestamps(bag_path)

    # Analyze timestamps
    analysis_data = analyze_timestamps(timestamps)

    # Create plots
    create_plots(analysis_data)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)