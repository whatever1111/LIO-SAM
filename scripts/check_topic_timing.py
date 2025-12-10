#!/usr/bin/env python3
"""
Check timestamp ordering of different topics in bag file.
"""

import rosbag

BAG_FILE = "/root/autodl-tmp/info_fixed.bag"

topics = [
    "/fixposition/fpa/corrimu",
    "/imu/data",
    "/fixposition/fpa/odometry",
    "/lidar_points"
]

def main():
    print("="*70)
    print("Topic Timestamp Analysis")
    print("="*70)

    bag = rosbag.Bag(BAG_FILE)

    first_times = {}
    for topic in topics:
        for t, msg, timestamp in bag.read_messages(topics=[topic]):
            first_times[topic] = timestamp.to_sec()
            break

    bag.close()

    if not first_times:
        print("No messages found!")
        return

    # Sort by timestamp
    sorted_topics = sorted(first_times.items(), key=lambda x: x[1])

    print("\nFirst message timestamps (sorted):")
    print("-"*70)

    base_time = sorted_topics[0][1]
    for topic, t in sorted_topics:
        rel_time = t - base_time
        print(f"  {topic:40s} t={t:.6f} (+{rel_time*1000:.3f} ms)")

    print("-"*70)

    # Check if /imu/data comes before /fixposition/fpa/corrimu
    imu_data_time = first_times.get("/imu/data", float('inf'))
    corrimu_time = first_times.get("/fixposition/fpa/corrimu", float('inf'))

    print(f"\n/imu/data vs CORRIMU:")
    if imu_data_time < corrimu_time:
        print(f"  /imu/data arrives {(corrimu_time - imu_data_time)*1000:.3f} ms BEFORE CORRIMU ✓")
    else:
        print(f"  CORRIMU arrives {(imu_data_time - corrimu_time)*1000:.3f} ms BEFORE /imu/data ✗")
        print(f"  This can cause hasImuOrientation_=false when first CORRIMU is processed!")


if __name__ == "__main__":
    main()
