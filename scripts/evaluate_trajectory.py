#!/usr/bin/env python3
"""
轨迹评估和可视化脚本
分析LIO-SAM融合轨迹与GPS的对比，特别是GNSS降级区间
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation
import sys
import os

class TrajectoryEvaluator:
    def __init__(self, bag_file, gnss_status_file):
        self.bag_file = bag_file
        self.gnss_status_file = gnss_status_file

        # Data storage
        self.fusion_data = []  # [(t, x, y, z), ...]
        self.gps_data = []
        self.degraded_intervals = []

        print(f"Loading data from {bag_file}...")
        self.load_data()
        self.load_gnss_status()

    def load_data(self):
        """Load trajectory data from bag file"""
        # Allow reading unindexed bags (in case of improper shutdown)
        bag = rosbag.Bag(self.bag_file, 'r', allow_unindexed=True)

        for topic, msg, t in bag.read_messages(topics=['/lio_sam/mapping/odometry']):
            self.fusion_data.append([
                msg.header.stamp.to_sec(),
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

        for topic, msg, t in bag.read_messages(topics=['/odometry/gps']):
            self.gps_data.append([
                msg.header.stamp.to_sec(),
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])

        bag.close()

        self.fusion_data = np.array(self.fusion_data)
        self.gps_data = np.array(self.gps_data)

        print(f"Loaded {len(self.fusion_data)} fusion poses")
        print(f"Loaded {len(self.gps_data)} GPS poses")

    def load_gnss_status(self):
        """Load GNSS degradation status"""
        if not os.path.exists(self.gnss_status_file):
            print(f"Warning: GNSS status file not found: {self.gnss_status_file}")
            return

        df = pd.read_csv(self.gnss_status_file)

        # Find degradation intervals
        degraded = df['is_degraded'].values
        timestamps = df['timestamp'].values

        in_degradation = False
        start_time = None

        for i, (is_deg, t) in enumerate(zip(degraded, timestamps)):
            if is_deg and not in_degradation:
                # Start of degradation
                start_time = t
                in_degradation = True
            elif not is_deg and in_degradation:
                # End of degradation
                self.degraded_intervals.append((start_time, t))
                in_degradation = False

        # If still degraded at end
        if in_degradation:
            self.degraded_intervals.append((start_time, timestamps[-1]))

        print(f"Found {len(self.degraded_intervals)} GNSS degradation intervals:")
        for i, (start, end) in enumerate(self.degraded_intervals):
            print(f"  Interval {i+1}: {start:.3f} - {end:.3f} ({end-start:.3f}s)")

    def interpolate_trajectory(self, traj_query, traj_ref):
        """Interpolate trajectory to match timestamps"""
        t_query = traj_query[:, 0]
        t_ref = traj_ref[:, 0]

        # Find common time range
        t_start = max(t_query[0], t_ref[0])
        t_end = min(t_query[-1], t_ref[-1])

        # Interpolate on reference timestamps
        indices = (t_ref >= t_start) & (t_ref <= t_end)
        t_common = t_ref[indices]

        query_interp = np.zeros((len(t_common), 3))
        for i in range(3):
            query_interp[:, i] = np.interp(t_common, t_query, traj_query[:, i+1])

        ref_interp = traj_ref[indices, 1:4]

        return t_common, query_interp, ref_interp

    def compute_errors(self):
        """Compute position errors between fusion and GPS"""
        t_common, fusion_interp, gps_interp = self.interpolate_trajectory(
            self.fusion_data, self.gps_data
        )

        # Compute errors
        errors = np.linalg.norm(fusion_interp - gps_interp, axis=1)

        return t_common, errors, fusion_interp, gps_interp

    def plot_trajectories_3d(self, output_file='trajectory_3d.png'):
        """Plot 3D trajectories"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot full trajectories
        ax.plot(self.gps_data[:, 1], self.gps_data[:, 2], self.gps_data[:, 3],
                'g-', label='GPS', linewidth=2, alpha=0.7)
        ax.plot(self.fusion_data[:, 1], self.fusion_data[:, 2], self.fusion_data[:, 3],
                'b-', label='Fusion (LIO-SAM)', linewidth=2, alpha=0.7)

        # Highlight degraded intervals
        for start, end in self.degraded_intervals:
            # Find indices
            idx_fusion = (self.fusion_data[:, 0] >= start) & (self.fusion_data[:, 0] <= end)
            if np.any(idx_fusion):
                ax.plot(self.fusion_data[idx_fusion, 1],
                       self.fusion_data[idx_fusion, 2],
                       self.fusion_data[idx_fusion, 3],
                       'r-', linewidth=3, alpha=0.8)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory Comparison (Red = GNSS Degraded)')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved 3D trajectory plot to {output_file}")
        plt.close()

    def plot_trajectories_2d(self, output_file='trajectory_2d.png'):
        """Plot 2D top-down view"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot full trajectories
        ax.plot(self.gps_data[:, 1], self.gps_data[:, 2],
                'g-', label='GPS', linewidth=2, alpha=0.7)
        ax.plot(self.fusion_data[:, 1], self.fusion_data[:, 2],
                'b-', label='Fusion (LIO-SAM)', linewidth=2, alpha=0.7)

        # Highlight degraded intervals
        for i, (start, end) in enumerate(self.degraded_intervals):
            idx_fusion = (self.fusion_data[:, 0] >= start) & (self.fusion_data[:, 0] <= end)
            if np.any(idx_fusion):
                label = 'GNSS Degraded' if i == 0 else None
                ax.plot(self.fusion_data[idx_fusion, 1],
                       self.fusion_data[idx_fusion, 2],
                       'r-', linewidth=3, alpha=0.8, label=label)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('2D Trajectory Comparison (Top View)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved 2D trajectory plot to {output_file}")
        plt.close()

    def plot_errors(self, output_file='errors.png'):
        """Plot position errors over time"""
        t_common, errors, _, _ = self.compute_errors()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot errors
        ax1.plot(t_common - t_common[0], errors, 'b-', linewidth=1.5, alpha=0.7)

        # Highlight degraded intervals
        for start, end in self.degraded_intervals:
            ax1.axvspan(start - t_common[0], end - t_common[0],
                       color='red', alpha=0.2, label='GNSS Degraded')

        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Position Error (m)', fontsize=12)
        ax1.set_title('Position Error: Fusion vs GPS', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot cumulative error distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax2.plot(sorted_errors, cumulative, 'b-', linewidth=2)
        ax2.set_xlabel('Position Error (m)', fontsize=12)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.set_title('Cumulative Error Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Mark percentiles
        for percentile in [50, 90, 95, 99]:
            value = np.percentile(errors, percentile)
            ax2.axvline(value, color='r', linestyle='--', alpha=0.5)
            ax2.text(value, percentile, f'{percentile}%: {value:.3f}m',
                    rotation=90, va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved error plot to {output_file}")
        plt.close()

    def plot_degraded_regions(self, output_dir='degraded_regions'):
        """Plot detailed views of degraded regions"""
        os.makedirs(output_dir, exist_ok=True)

        t_common, errors, fusion_interp, gps_interp = self.compute_errors()

        for i, (start, end) in enumerate(self.degraded_intervals):
            # Add buffer before and after
            buffer = 10.0  # seconds
            start_buffered = start - buffer
            end_buffered = end + buffer

            # Find indices
            idx = (t_common >= start_buffered) & (t_common <= end_buffered)
            idx_degraded = (t_common >= start) & (t_common <= end)

            if not np.any(idx):
                continue

            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 2D trajectory
            ax1.plot(gps_interp[idx, 0], gps_interp[idx, 1],
                    'g-', label='GPS', linewidth=2, alpha=0.7)
            ax1.plot(fusion_interp[idx, 0], fusion_interp[idx, 1],
                    'b-', label='Fusion', linewidth=2, alpha=0.7)
            ax1.plot(fusion_interp[idx_degraded, 0], fusion_interp[idx_degraded, 1],
                    'r-', label='Degraded Period', linewidth=3, alpha=0.8)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'Degraded Region {i+1}: Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Error over time
            t_plot = t_common[idx] - t_common[idx][0]
            ax2.plot(t_plot, errors[idx], 'b-', linewidth=2)
            degraded_mask = idx_degraded & idx
            ax2.axvspan(start - t_common[idx][0], end - t_common[idx][0],
                       color='red', alpha=0.2, label='GNSS Degraded')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position Error (m)')
            ax2.set_title(f'Degraded Region {i+1}: Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'degraded_region_{i+1}.png')
            plt.savefig(output_file, dpi=300)
            print(f"Saved degraded region {i+1} plot to {output_file}")
            plt.close()

    def compute_statistics(self, output_file='statistics.txt'):
        """Compute and save error statistics"""
        t_common, errors, _, _ = self.compute_errors()

        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LIO-SAM GNSS Fusion Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # Global statistics
            f.write("Global Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total trajectory points: {len(errors)}\n")
            f.write(f"Mean error: {np.mean(errors):.4f} m\n")
            f.write(f"Std dev: {np.std(errors):.4f} m\n")
            f.write(f"RMS error: {np.sqrt(np.mean(errors**2)):.4f} m\n")
            f.write(f"Max error: {np.max(errors):.4f} m\n")
            f.write(f"Median error: {np.median(errors):.4f} m\n")
            f.write(f"50th percentile: {np.percentile(errors, 50):.4f} m\n")
            f.write(f"90th percentile: {np.percentile(errors, 90):.4f} m\n")
            f.write(f"95th percentile: {np.percentile(errors, 95):.4f} m\n")
            f.write(f"99th percentile: {np.percentile(errors, 99):.4f} m\n\n")

            # Degraded region statistics
            f.write("GNSS Degraded Region Statistics:\n")
            f.write("-" * 60 + "\n")

            for i, (start, end) in enumerate(self.degraded_intervals):
                idx = (t_common >= start) & (t_common <= end)
                if not np.any(idx):
                    continue

                errors_deg = errors[idx]
                duration = end - start

                f.write(f"\nRegion {i+1}:\n")
                f.write(f"  Duration: {duration:.2f} s\n")
                f.write(f"  Points: {len(errors_deg)}\n")
                f.write(f"  Mean error: {np.mean(errors_deg):.4f} m\n")
                f.write(f"  RMS error: {np.sqrt(np.mean(errors_deg**2)):.4f} m\n")
                f.write(f"  Max error: {np.max(errors_deg):.4f} m\n")
                f.write(f"  Median error: {np.median(errors_deg):.4f} m\n")

            # Non-degraded statistics
            f.write("\n" + "=" * 60 + "\n")
            f.write("Non-Degraded Region Statistics:\n")
            f.write("-" * 60 + "\n")

            # Create mask for non-degraded regions
            non_degraded_mask = np.ones(len(t_common), dtype=bool)
            for start, end in self.degraded_intervals:
                idx = (t_common >= start) & (t_common <= end)
                non_degraded_mask &= ~idx

            if np.any(non_degraded_mask):
                errors_nondeg = errors[non_degraded_mask]
                f.write(f"Points: {len(errors_nondeg)}\n")
                f.write(f"Mean error: {np.mean(errors_nondeg):.4f} m\n")
                f.write(f"RMS error: {np.sqrt(np.mean(errors_nondeg**2)):.4f} m\n")
                f.write(f"Max error: {np.max(errors_nondeg):.4f} m\n")
                f.write(f"Median error: {np.median(errors_nondeg):.4f} m\n")

        print(f"Saved statistics to {output_file}")

        # Also print to console
        with open(output_file, 'r') as f:
            print("\n" + f.read())

    def run_evaluation(self, output_dir='.'):
        """Run complete evaluation"""
        print("\n" + "=" * 60)
        print("Starting Trajectory Evaluation")
        print("=" * 60 + "\n")

        os.makedirs(output_dir, exist_ok=True)

        self.plot_trajectories_3d(os.path.join(output_dir, 'trajectory_3d.png'))
        self.plot_trajectories_2d(os.path.join(output_dir, 'trajectory_2d.png'))
        self.plot_errors(os.path.join(output_dir, 'errors.png'))
        self.plot_degraded_regions(os.path.join(output_dir, 'degraded_regions'))
        self.compute_statistics(os.path.join(output_dir, 'statistics.txt'))

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 evaluate_trajectory.py <bag_file> <gnss_status_file> [output_dir]")
        sys.exit(1)

    bag_file = sys.argv[1]
    gnss_status_file = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './evaluation_results'

    evaluator = TrajectoryEvaluator(bag_file, gnss_status_file)
    evaluator.run_evaluation(output_dir)
