#!/usr/bin/env python3
"""
Real-time GPS and LIO-SAM Covariance Monitor

Subscribes to live ROS topics and displays/logs covariance statistics in real-time.
Can also record data to CSV for later analysis.

Usage:
  rosrun lio_sam monitor_covariance.py
  rosrun lio_sam monitor_covariance.py --record  # Record to CSV
"""

import rospy
import numpy as np
import threading
import time
import os
from datetime import datetime
from collections import deque

from nav_msgs.msg import Odometry

# Try to import fixposition messages
try:
    from fixposition_driver_msgs.msg import FpaOdometry, FpaOdomenu
    HAS_FPA_MSGS = True
except ImportError:
    HAS_FPA_MSGS = False
    rospy.logwarn("fixposition_driver_msgs not found, using nav_msgs/Odometry only")


class CovarianceMonitor:
    def __init__(self, record=False, output_dir=None):
        self.record = record
        self.output_dir = output_dir or '/root/autodl-tmp/catkin_ws/src/LIO-SAM/output'

        # Data storage (rolling window)
        self.window_size = 100  # Keep last 100 messages for statistics
        self.data = {
            'gps_odometry': {
                'times': deque(maxlen=self.window_size),
                'pos_cov': deque(maxlen=self.window_size),
                'ori_cov': deque(maxlen=self.window_size),
                'count': 0
            },
            'gps_odomenu': {
                'times': deque(maxlen=self.window_size),
                'pos_cov': deque(maxlen=self.window_size),
                'ori_cov': deque(maxlen=self.window_size),
                'count': 0
            },
            'lio_odometry': {
                'times': deque(maxlen=self.window_size),
                'pos_cov': deque(maxlen=self.window_size),
                'ori_cov': deque(maxlen=self.window_size),
                'count': 0
            }
        }

        # CSV files for recording
        self.csv_files = {}
        if self.record:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for source in self.data.keys():
                path = os.path.join(self.output_dir, f'covariance_{source}_{timestamp}.csv')
                self.csv_files[source] = open(path, 'w')
                self.csv_files[source].write('timestamp,pos_cov_x,pos_cov_y,pos_cov_z,ori_cov_r,ori_cov_p,ori_cov_y\n')
                rospy.loginfo(f"Recording {source} to {path}")

        # Lock for thread safety
        self.lock = threading.Lock()

        # Subscribe to topics
        self.subscribers = []

        # GPS Odometry (FPA format)
        if HAS_FPA_MSGS:
            self.subscribers.append(
                rospy.Subscriber('/fixposition/fpa/odometry', FpaOdometry,
                               self.gps_odometry_callback, queue_size=10)
            )
            self.subscribers.append(
                rospy.Subscriber('/fixposition/fpa/odomenu', FpaOdomenu,
                               self.gps_odomenu_callback, queue_size=10)
            )
        else:
            # Fallback to nav_msgs/Odometry
            self.subscribers.append(
                rospy.Subscriber('/odometry/gps', Odometry,
                               self.gps_odom_nav_callback, queue_size=10)
            )

        # LIO-SAM Odometry
        self.subscribers.append(
            rospy.Subscriber('/lio_sam/mapping/odometry', Odometry,
                           self.lio_odometry_callback, queue_size=10)
        )

        rospy.loginfo("Covariance Monitor started")

    def extract_covariance(self, msg, is_fpa=False):
        """Extract position and orientation covariance from message"""
        if is_fpa:
            cov = np.array(msg.pose.covariance).reshape(6, 6)
        else:
            cov = np.array(msg.pose.covariance).reshape(6, 6)

        pos_cov = np.diag(cov[:3, :3])  # [x, y, z]
        ori_cov = np.diag(cov[3:, 3:])  # [roll, pitch, yaw]

        return pos_cov, ori_cov

    def gps_odometry_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            pos_cov, ori_cov = self.extract_covariance(msg, is_fpa=True)

            self.data['gps_odometry']['times'].append(t)
            self.data['gps_odometry']['pos_cov'].append(pos_cov)
            self.data['gps_odometry']['ori_cov'].append(ori_cov)
            self.data['gps_odometry']['count'] += 1

            if self.record and 'gps_odometry' in self.csv_files:
                self.csv_files['gps_odometry'].write(
                    f"{t},{pos_cov[0]},{pos_cov[1]},{pos_cov[2]},"
                    f"{ori_cov[0]},{ori_cov[1]},{ori_cov[2]}\n"
                )

    def gps_odomenu_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            pos_cov, ori_cov = self.extract_covariance(msg, is_fpa=True)

            self.data['gps_odomenu']['times'].append(t)
            self.data['gps_odomenu']['pos_cov'].append(pos_cov)
            self.data['gps_odomenu']['ori_cov'].append(ori_cov)
            self.data['gps_odomenu']['count'] += 1

            if self.record and 'gps_odomenu' in self.csv_files:
                self.csv_files['gps_odomenu'].write(
                    f"{t},{pos_cov[0]},{pos_cov[1]},{pos_cov[2]},"
                    f"{ori_cov[0]},{ori_cov[1]},{ori_cov[2]}\n"
                )

    def gps_odom_nav_callback(self, msg):
        """Fallback for nav_msgs/Odometry GPS"""
        with self.lock:
            t = msg.header.stamp.to_sec()
            pos_cov, ori_cov = self.extract_covariance(msg, is_fpa=False)

            self.data['gps_odomenu']['times'].append(t)
            self.data['gps_odomenu']['pos_cov'].append(pos_cov)
            self.data['gps_odomenu']['ori_cov'].append(ori_cov)
            self.data['gps_odomenu']['count'] += 1

    def lio_odometry_callback(self, msg):
        with self.lock:
            t = msg.header.stamp.to_sec()
            pos_cov, ori_cov = self.extract_covariance(msg, is_fpa=False)

            self.data['lio_odometry']['times'].append(t)
            self.data['lio_odometry']['pos_cov'].append(pos_cov)
            self.data['lio_odometry']['ori_cov'].append(ori_cov)
            self.data['lio_odometry']['count'] += 1

            if self.record and 'lio_odometry' in self.csv_files:
                self.csv_files['lio_odometry'].write(
                    f"{t},{pos_cov[0]},{pos_cov[1]},{pos_cov[2]},"
                    f"{ori_cov[0]},{ori_cov[1]},{ori_cov[2]}\n"
                )

    def compute_statistics(self, source):
        """Compute statistics for a source"""
        data = self.data[source]
        if len(data['pos_cov']) == 0:
            return None

        pos_covs = np.array(data['pos_cov'])
        ori_covs = np.array(data['ori_cov'])

        # Filter valid (positive) values
        valid_pos = pos_covs[np.all(pos_covs > 0, axis=1)]
        valid_ori = ori_covs[np.all(ori_covs > 0, axis=1)]

        stats = {
            'count': data['count'],
            'window': len(data['pos_cov']),
        }

        if len(valid_pos) > 0:
            stats['pos_mean_var'] = np.mean(valid_pos, axis=0)
            stats['pos_mean_std'] = np.sqrt(stats['pos_mean_var'])
            stats['pos_current'] = valid_pos[-1] if len(valid_pos) > 0 else None
        else:
            stats['pos_mean_var'] = None
            stats['pos_all_zero'] = True

        if len(valid_ori) > 0:
            stats['ori_mean_var'] = np.mean(valid_ori, axis=0)
            stats['ori_mean_std_deg'] = np.rad2deg(np.sqrt(stats['ori_mean_var']))
            stats['ori_current'] = valid_ori[-1] if len(valid_ori) > 0 else None
        else:
            stats['ori_mean_var'] = None
            stats['ori_all_zero'] = True

        return stats

    def print_status(self):
        """Print current status to terminal"""
        with self.lock:
            print("\033[2J\033[H")  # Clear screen
            print("=" * 70)
            print("GPS AND LIO-SAM COVARIANCE MONITOR")
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 70)

            for source in ['gps_odometry', 'gps_odomenu', 'lio_odometry']:
                stats = self.compute_statistics(source)
                print(f"\n{source.upper()}")
                print("-" * 40)

                if stats is None:
                    print("  No data received yet")
                    continue

                print(f"  Total msgs: {stats['count']}, Window: {stats['window']}")

                if stats.get('pos_mean_var') is not None:
                    print(f"  Position Std (mean):  [{stats['pos_mean_std'][0]:.4f}, "
                          f"{stats['pos_mean_std'][1]:.4f}, {stats['pos_mean_std'][2]:.4f}] m")
                    if stats.get('pos_current') is not None:
                        cur_std = np.sqrt(stats['pos_current'])
                        print(f"  Position Std (current): [{cur_std[0]:.4f}, "
                              f"{cur_std[1]:.4f}, {cur_std[2]:.4f}] m")
                else:
                    print("  Position Cov: ALL ZERO (not provided)")

                if stats.get('ori_mean_var') is not None:
                    print(f"  Orientation Std (mean): [{stats['ori_mean_std_deg'][0]:.2f}, "
                          f"{stats['ori_mean_std_deg'][1]:.2f}, {stats['ori_mean_std_deg'][2]:.2f}] deg")
                else:
                    print("  Orientation Cov: ALL ZERO (not provided)")

            print("\n" + "=" * 70)
            if self.record:
                print("Recording to CSV files...")
            print("Press Ctrl+C to exit")

    def run(self):
        """Main loop"""
        rate = rospy.Rate(2)  # 2 Hz update

        while not rospy.is_shutdown():
            self.print_status()
            rate.sleep()

    def shutdown(self):
        """Cleanup on shutdown"""
        for f in self.csv_files.values():
            f.close()
        rospy.loginfo("Covariance monitor stopped")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor GPS and LIO-SAM covariances')
    parser.add_argument('--record', '-r', action='store_true', help='Record to CSV files')
    parser.add_argument('--output', '-o', default='/root/autodl-tmp/catkin_ws/src/LIO-SAM/output',
                       help='Output directory for CSV files')

    # Parse args before rospy.init_node to allow --help
    args, unknown = parser.parse_known_args()

    rospy.init_node('covariance_monitor', anonymous=True)

    monitor = CovarianceMonitor(record=args.record, output_dir=args.output)
    rospy.on_shutdown(monitor.shutdown)

    try:
        monitor.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
