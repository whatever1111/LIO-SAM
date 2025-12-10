#!/usr/bin/env python3
"""
GPS and LIO-SAM Covariance Analyzer

This script analyzes and visualizes:
1. GPS input covariance (from /fixposition/fpa/odometry or /fixposition/fpa/odomenu)
2. LIO-SAM output covariance (from /lio_sam/mapping/odometry with covariance)

Can work with:
- Live ROS topics
- Recorded bag files
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from datetime import datetime

try:
    import rosbag
except ImportError:
    print("ERROR: rosbag not found. Run: source /opt/ros/noetic/setup.bash")
    sys.exit(1)


def extract_covariance_from_bag(bag_path, max_messages=None):
    """Extract GPS and LIO covariance data from bag file"""

    print(f"Opening bag file: {bag_path}")
    bag = rosbag.Bag(bag_path)

    # Data storage
    data = {
        'gps_odometry': {  # /fixposition/fpa/odometry
            'times': [], 'pos': [], 'ori': [],
            'pos_cov': [], 'ori_cov': [], 'full_cov': []
        },
        'gps_odomenu': {  # /fixposition/fpa/odomenu (ENU frame)
            'times': [], 'pos': [],
            'pos_cov': [], 'ori_cov': [], 'full_cov': []
        },
        'lio_odometry': {  # /lio_sam/mapping/odometry
            'times': [], 'pos': [], 'ori': [],
            'pos_cov': [], 'ori_cov': [], 'full_cov': [],
            'is_degenerate': []
        },
        'lio_incremental': {  # /lio_sam/mapping/odometry_incremental
            'times': [], 'pos': [], 'ori': [],
            'is_degenerate': []
        }
    }

    # Topic mapping
    topics_of_interest = [
        '/fixposition/fpa/odometry',
        '/fixposition/fpa/odomenu',
        '/lio_sam/mapping/odometry',
        '/lio_sam/mapping/odometry_incremental',
        '/odometry/gps'
    ]

    msg_counts = defaultdict(int)

    print("Reading messages...")
    for topic, msg, t in bag.read_messages(topics=topics_of_interest):
        msg_counts[topic] += 1

        if max_messages and msg_counts[topic] > max_messages:
            continue

        timestamp = t.to_sec()

        # Extract pose and covariance based on message type
        if topic == '/fixposition/fpa/odometry':
            store = data['gps_odometry']
            store['times'].append(timestamp)

            # FpaOdometry has pose as PoseWithCovariance (not nested in pose.pose)
            if hasattr(msg, 'pose'):
                if hasattr(msg.pose, 'pose'):
                    # nav_msgs/Odometry style
                    pos = msg.pose.pose.position
                    ori = msg.pose.pose.orientation
                    cov = np.array(msg.pose.covariance).reshape(6, 6)
                elif hasattr(msg.pose, 'position'):
                    # PoseWithCovariance style
                    pos = msg.pose.position
                    ori = msg.pose.orientation
                    cov = np.array(msg.pose.covariance).reshape(6, 6)
                else:
                    continue

                store['pos'].append([pos.x, pos.y, pos.z])
                store['ori'].append([ori.x, ori.y, ori.z, ori.w])
                store['full_cov'].append(cov)
                store['pos_cov'].append(cov[:3, :3])
                store['ori_cov'].append(cov[3:, 3:])

        elif topic == '/fixposition/fpa/odomenu' or topic == '/odometry/gps':
            store = data['gps_odomenu']
            store['times'].append(timestamp)

            if hasattr(msg, 'pose'):
                if hasattr(msg.pose, 'pose'):
                    pos = msg.pose.pose.position
                    cov = np.array(msg.pose.covariance).reshape(6, 6)
                elif hasattr(msg.pose, 'position'):
                    pos = msg.pose.position
                    cov = np.array(msg.pose.covariance).reshape(6, 6)
                else:
                    continue

                store['pos'].append([pos.x, pos.y, pos.z])
                store['full_cov'].append(cov)
                store['pos_cov'].append(cov[:3, :3])
                store['ori_cov'].append(cov[3:, 3:])

        elif topic == '/lio_sam/mapping/odometry':
            store = data['lio_odometry']
            store['times'].append(timestamp)

            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            cov = np.array(msg.pose.covariance).reshape(6, 6)

            store['pos'].append([pos.x, pos.y, pos.z])
            store['ori'].append([ori.x, ori.y, ori.z, ori.w])
            store['full_cov'].append(cov)
            store['pos_cov'].append(cov[:3, :3])
            store['ori_cov'].append(cov[3:, 3:])

        elif topic == '/lio_sam/mapping/odometry_incremental':
            store = data['lio_incremental']
            store['times'].append(timestamp)

            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            cov = np.array(msg.pose.covariance).reshape(6, 6)

            store['pos'].append([pos.x, pos.y, pos.z])
            store['ori'].append([ori.x, ori.y, ori.z, ori.w])
            # cov[0] is used as degenerate flag in LIO-SAM
            store['is_degenerate'].append(cov[0, 0] == 1)

    bag.close()

    print(f"\nMessage counts:")
    for topic, count in sorted(msg_counts.items()):
        print(f"  {topic}: {count}")

    return data


def compute_statistics(data):
    """Compute statistics for covariance data"""

    stats = {}

    for source, source_data in data.items():
        if not source_data['times']:
            continue

        stats[source] = {
            'count': len(source_data['times']),
            'duration': source_data['times'][-1] - source_data['times'][0] if len(source_data['times']) > 1 else 0,
        }

        # Position covariance statistics
        if source_data.get('pos_cov'):
            pos_covs = np.array(source_data['pos_cov'])
            pos_diag = np.array([np.diag(c) for c in pos_covs])

            # Filter valid (positive) covariances
            valid_mask = np.all(pos_diag > 0, axis=1)

            stats[source]['pos_cov'] = {
                'valid_count': np.sum(valid_mask),
                'valid_ratio': np.mean(valid_mask),
                'all_zero': np.allclose(pos_diag, 0),
            }

            if np.sum(valid_mask) > 0:
                valid_diag = pos_diag[valid_mask]
                stats[source]['pos_cov'].update({
                    'mean': np.mean(valid_diag, axis=0),
                    'std': np.std(valid_diag, axis=0),
                    'min': np.min(valid_diag, axis=0),
                    'max': np.max(valid_diag, axis=0),
                    'median': np.median(valid_diag, axis=0),
                    # Convert variance to standard deviation for intuition
                    'mean_std_dev': np.sqrt(np.mean(valid_diag, axis=0)),
                })

        # Orientation covariance statistics
        if source_data.get('ori_cov'):
            ori_covs = np.array(source_data['ori_cov'])
            ori_diag = np.array([np.diag(c) for c in ori_covs])

            valid_mask = np.all(ori_diag > 0, axis=1)

            stats[source]['ori_cov'] = {
                'valid_count': np.sum(valid_mask),
                'valid_ratio': np.mean(valid_mask),
                'all_zero': np.allclose(ori_diag, 0),
            }

            if np.sum(valid_mask) > 0:
                valid_diag = ori_diag[valid_mask]
                stats[source]['ori_cov'].update({
                    'mean': np.mean(valid_diag, axis=0),
                    'std': np.std(valid_diag, axis=0),
                    'min': np.min(valid_diag, axis=0),
                    'max': np.max(valid_diag, axis=0),
                    'median': np.median(valid_diag, axis=0),
                    # Convert to degrees for intuition
                    'mean_std_dev_deg': np.rad2deg(np.sqrt(np.mean(valid_diag, axis=0))),
                })

    return stats


def print_statistics(stats):
    """Print formatted statistics"""

    print("\n" + "="*80)
    print("GPS AND LIO-SAM COVARIANCE STATISTICS")
    print("="*80)

    for source, s in stats.items():
        print(f"\n{'─'*60}")
        print(f"Source: {source}")
        print(f"{'─'*60}")
        print(f"  Messages: {s['count']}, Duration: {s['duration']:.1f}s")

        if 'pos_cov' in s:
            pc = s['pos_cov']
            print(f"\n  Position Covariance:")
            print(f"    Valid: {pc['valid_count']} ({pc['valid_ratio']*100:.1f}%)")
            if not pc['all_zero'] and pc['valid_count'] > 0:
                print(f"    Mean variance [X,Y,Z]: [{pc['mean'][0]:.4f}, {pc['mean'][1]:.4f}, {pc['mean'][2]:.4f}] m²")
                print(f"    Mean std dev  [X,Y,Z]: [{pc['mean_std_dev'][0]:.4f}, {pc['mean_std_dev'][1]:.4f}, {pc['mean_std_dev'][2]:.4f}] m")
                print(f"    Min  variance [X,Y,Z]: [{pc['min'][0]:.6f}, {pc['min'][1]:.6f}, {pc['min'][2]:.6f}] m²")
                print(f"    Max  variance [X,Y,Z]: [{pc['max'][0]:.4f}, {pc['max'][1]:.4f}, {pc['max'][2]:.4f}] m²")
            else:
                print(f"    ALL ZERO - no covariance provided")

        if 'ori_cov' in s:
            oc = s['ori_cov']
            print(f"\n  Orientation Covariance:")
            print(f"    Valid: {oc['valid_count']} ({oc['valid_ratio']*100:.1f}%)")
            if not oc['all_zero'] and oc['valid_count'] > 0:
                print(f"    Mean variance [R,P,Y]: [{oc['mean'][0]:.6f}, {oc['mean'][1]:.6f}, {oc['mean'][2]:.6f}] rad²")
                print(f"    Mean std dev  [R,P,Y]: [{oc['mean_std_dev_deg'][0]:.2f}, {oc['mean_std_dev_deg'][1]:.2f}, {oc['mean_std_dev_deg'][2]:.2f}] deg")
                print(f"    Min  variance [R,P,Y]: [{oc['min'][0]:.6f}, {oc['min'][1]:.6f}, {oc['min'][2]:.6f}] rad²")
                print(f"    Max  variance [R,P,Y]: [{oc['max'][0]:.6f}, {oc['max'][1]:.6f}, {oc['max'][2]:.6f}] rad²")
            else:
                print(f"    ALL ZERO - no covariance provided")


def visualize_covariances(data, stats, output_dir):
    """Generate comprehensive visualization"""

    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)

    fig.suptitle('GPS vs LIO-SAM Covariance Analysis', fontsize=14, fontweight='bold')

    # Get reference time
    all_times = []
    for source_data in data.values():
        if source_data['times']:
            all_times.extend(source_data['times'])
    t0 = min(all_times) if all_times else 0

    colors = {'gps': '#2ecc71', 'lio': '#3498db', 'gps_enu': '#e74c3c'}

    # Row 0: GPS Odometry Position Covariance
    if data['gps_odometry']['times'] and data['gps_odometry']['pos_cov']:
        t = np.array(data['gps_odometry']['times']) - t0
        pos_cov = np.array(data['gps_odometry']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])

        ax = fig.add_subplot(gs[0, 0])
        ax.semilogy(t, pos_diag[:, 0], 'b-', alpha=0.7, label='X')
        ax.semilogy(t, pos_diag[:, 1], 'g-', alpha=0.7, label='Y')
        ax.semilogy(t, pos_diag[:, 2], 'r-', alpha=0.7, label='Z')
        ax.set_title('GPS Odometry - Position Variance')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Variance (m²)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Orientation covariance
        if data['gps_odometry']['ori_cov']:
            ori_cov = np.array(data['gps_odometry']['ori_cov'])
            ori_diag = np.array([np.diag(c) for c in ori_cov])

            ax = fig.add_subplot(gs[0, 1])
            ax.semilogy(t, np.rad2deg(np.sqrt(ori_diag[:, 0])), 'b-', alpha=0.7, label='Roll')
            ax.semilogy(t, np.rad2deg(np.sqrt(ori_diag[:, 1])), 'g-', alpha=0.7, label='Pitch')
            ax.semilogy(t, np.rad2deg(np.sqrt(ori_diag[:, 2])), 'r-', alpha=0.7, label='Yaw')
            ax.set_title('GPS Odometry - Orientation Std Dev')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Std Dev (deg)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

    # Row 0, Col 2: GPS ENU Position Covariance
    if data['gps_odomenu']['times'] and data['gps_odomenu']['pos_cov']:
        t = np.array(data['gps_odomenu']['times']) - t0
        pos_cov = np.array(data['gps_odomenu']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])

        ax = fig.add_subplot(gs[0, 2])
        ax.semilogy(t, pos_diag[:, 0], 'b-', alpha=0.7, label='E')
        ax.semilogy(t, pos_diag[:, 1], 'g-', alpha=0.7, label='N')
        ax.semilogy(t, pos_diag[:, 2], 'r-', alpha=0.7, label='U')
        ax.set_title('GPS OdomENU - Position Variance')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Variance (m²)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Row 1: LIO-SAM Odometry Covariance (if available)
    if data['lio_odometry']['times'] and data['lio_odometry']['pos_cov']:
        t = np.array(data['lio_odometry']['times']) - t0
        pos_cov = np.array(data['lio_odometry']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])

        # Check if LIO publishes real covariance
        if not np.allclose(pos_diag, 0):
            ax = fig.add_subplot(gs[1, 0])
            ax.semilogy(t, pos_diag[:, 0], 'b-', alpha=0.7, label='X')
            ax.semilogy(t, pos_diag[:, 1], 'g-', alpha=0.7, label='Y')
            ax.semilogy(t, pos_diag[:, 2], 'r-', alpha=0.7, label='Z')
            ax.set_title('LIO-SAM - Position Variance')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Variance (m²)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax = fig.add_subplot(gs[1, 0])
            ax.text(0.5, 0.5, 'LIO-SAM does not publish\nposition covariance\n(all zeros)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('LIO-SAM - Position Variance')
    else:
        ax = fig.add_subplot(gs[1, 0])
        ax.text(0.5, 0.5, 'No LIO-SAM odometry data',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('LIO-SAM - Position Variance')

    # Row 1, Col 1-2: Comparison histograms
    ax = fig.add_subplot(gs[1, 1])
    hist_data = []
    hist_labels = []
    hist_colors = []

    if data['gps_odometry']['pos_cov']:
        pos_cov = np.array(data['gps_odometry']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])
        valid = pos_diag[pos_diag[:, 0] > 0]
        if len(valid) > 0:
            # Use standard deviation for histogram
            hist_data.append(np.sqrt(valid[:, 0]))  # X std dev
            hist_labels.append('GPS X')
            hist_colors.append('#2ecc71')

    if data['gps_odomenu']['pos_cov']:
        pos_cov = np.array(data['gps_odomenu']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])
        valid = pos_diag[pos_diag[:, 0] > 0]
        if len(valid) > 0:
            hist_data.append(np.sqrt(valid[:, 0]))
            hist_labels.append('GPS ENU E')
            hist_colors.append('#e74c3c')

    if hist_data:
        ax.hist(hist_data, bins=50, alpha=0.7, label=hist_labels, color=hist_colors[:len(hist_data)])
        ax.set_title('Position Std Dev Distribution')
        ax.set_xlabel('Std Dev (m)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Time-aligned comparison (GPS vs trajectory)
    ax = fig.add_subplot(gs[2, :2])

    # Plot GPS position uncertainty as error band
    if data['gps_odomenu']['times'] and data['gps_odomenu']['pos']:
        t = np.array(data['gps_odomenu']['times']) - t0
        pos = np.array(data['gps_odomenu']['pos'])
        pos_cov = np.array(data['gps_odomenu']['pos_cov'])
        pos_std = np.sqrt(np.array([np.diag(c) for c in pos_cov]))

        # Plot X position with uncertainty band
        ax.plot(t, pos[:, 0], 'g-', alpha=0.8, label='GPS E position', linewidth=1)
        ax.fill_between(t, pos[:, 0] - pos_std[:, 0], pos[:, 0] + pos_std[:, 0],
                       alpha=0.3, color='green', label='GPS ±1σ')

    if data['lio_odometry']['times'] and data['lio_odometry']['pos']:
        t = np.array(data['lio_odometry']['times']) - t0
        pos = np.array(data['lio_odometry']['pos'])
        ax.plot(t, pos[:, 0], 'b-', alpha=0.8, label='LIO-SAM X position', linewidth=1)

    ax.set_title('Position X/E with GPS Uncertainty Band')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2, Col 2: Covariance correlation matrix (GPS)
    if data['gps_odometry']['full_cov']:
        ax = fig.add_subplot(gs[2, 2])
        # Average covariance matrix
        avg_cov = np.mean(data['gps_odometry']['full_cov'], axis=0)
        # Convert to correlation matrix
        d = np.sqrt(np.diag(avg_cov))
        d[d == 0] = 1  # Avoid division by zero
        corr = avg_cov / np.outer(d, d)

        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('GPS Avg Correlation Matrix')
        labels = ['X', 'Y', 'Z', 'R', 'P', 'Yaw']
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Row 3: XY trajectory with covariance ellipses
    ax = fig.add_subplot(gs[3, :2])

    if data['gps_odomenu']['pos']:
        pos = np.array(data['gps_odomenu']['pos'])
        ax.plot(pos[:, 0], pos[:, 1], 'g-', alpha=0.5, label='GPS ENU', linewidth=1)

        # Draw covariance ellipses at intervals
        if data['gps_odomenu']['pos_cov']:
            pos_cov = np.array(data['gps_odomenu']['pos_cov'])
            step = max(1, len(pos) // 20)
            for i in range(0, len(pos), step):
                cov_2d = pos_cov[i][:2, :2]
                if np.all(np.diag(cov_2d) > 0):
                    draw_covariance_ellipse(ax, pos[i, :2], cov_2d, color='green', alpha=0.3)

    if data['lio_odometry']['pos']:
        pos = np.array(data['lio_odometry']['pos'])
        ax.plot(pos[:, 0], pos[:, 1], 'b-', alpha=0.8, label='LIO-SAM', linewidth=1)

    ax.set_title('XY Trajectory with GPS Covariance Ellipses')
    ax.set_xlabel('X/E (m)')
    ax.set_ylabel('Y/N (m)')
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # Row 3, Col 2: Statistics summary text
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')

    summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"

    for source in ['gps_odometry', 'gps_odomenu']:
        if source in stats:
            s = stats[source]
            name = "GPS Odometry" if source == 'gps_odometry' else "GPS OdomENU"
            summary_text += f"{name}:\n"
            summary_text += f"  Messages: {s['count']}\n"
            if 'pos_cov' in s and s['pos_cov']['valid_count'] > 0:
                pc = s['pos_cov']
                summary_text += f"  Pos std (mean): [{pc['mean_std_dev'][0]:.3f}, {pc['mean_std_dev'][1]:.3f}, {pc['mean_std_dev'][2]:.3f}] m\n"
            if 'ori_cov' in s and s['ori_cov']['valid_count'] > 0:
                oc = s['ori_cov']
                summary_text += f"  Ori std (mean): [{oc['mean_std_dev_deg'][0]:.2f}, {oc['mean_std_dev_deg'][1]:.2f}, {oc['mean_std_dev_deg'][2]:.2f}] deg\n"
            summary_text += "\n"

    if 'lio_odometry' in stats:
        s = stats['lio_odometry']
        summary_text += f"LIO-SAM Odometry:\n"
        summary_text += f"  Messages: {s['count']}\n"
        if 'pos_cov' in s:
            if s['pos_cov']['all_zero']:
                summary_text += f"  Pos cov: NOT PUBLISHED (all zero)\n"
            elif s['pos_cov']['valid_count'] > 0:
                pc = s['pos_cov']
                summary_text += f"  Pos std: [{pc['mean_std_dev'][0]:.3f}, {pc['mean_std_dev'][1]:.3f}, {pc['mean_std_dev'][2]:.3f}] m\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save figure
    output_path = os.path.join(output_dir, 'gps_lio_covariance_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def draw_covariance_ellipse(ax, center, cov, color='blue', alpha=0.3, n_std=2):
    """Draw a covariance ellipse"""
    from matplotlib.patches import Ellipse

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Compute angle
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Compute width and height (2 * sqrt(eigenvalue) for 1 std)
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])

    # Limit size for visualization
    max_size = 50
    width = min(width, max_size)
    height = min(height, max_size)

    ellipse = Ellipse(center, width, height, angle=angle,
                     facecolor=color, alpha=alpha, edgecolor=color)
    ax.add_patch(ellipse)


def generate_report(data, stats, output_dir):
    """Generate markdown report"""

    report = []
    report.append("# GPS and LIO-SAM Covariance Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Data Summary\n")
    report.append("| Source | Messages | Duration (s) | Rate (Hz) |")
    report.append("|--------|----------|--------------|-----------|")

    for source, s in stats.items():
        rate = s['count'] / s['duration'] if s['duration'] > 0 else 0
        report.append(f"| {source} | {s['count']} | {s['duration']:.1f} | {rate:.1f} |")

    report.append("\n## Position Covariance Statistics\n")
    report.append("| Source | Valid % | Mean Std X (m) | Mean Std Y (m) | Mean Std Z (m) |")
    report.append("|--------|---------|----------------|----------------|----------------|")

    for source, s in stats.items():
        if 'pos_cov' in s and s['pos_cov']['valid_count'] > 0:
            pc = s['pos_cov']
            report.append(f"| {source} | {pc['valid_ratio']*100:.1f} | {pc['mean_std_dev'][0]:.4f} | {pc['mean_std_dev'][1]:.4f} | {pc['mean_std_dev'][2]:.4f} |")
        elif 'pos_cov' in s:
            report.append(f"| {source} | 0 | N/A | N/A | N/A |")

    report.append("\n## Orientation Covariance Statistics\n")
    report.append("| Source | Valid % | Mean Std Roll (deg) | Mean Std Pitch (deg) | Mean Std Yaw (deg) |")
    report.append("|--------|---------|---------------------|----------------------|--------------------|")

    for source, s in stats.items():
        if 'ori_cov' in s and s['ori_cov']['valid_count'] > 0:
            oc = s['ori_cov']
            report.append(f"| {source} | {oc['valid_ratio']*100:.1f} | {oc['mean_std_dev_deg'][0]:.2f} | {oc['mean_std_dev_deg'][1]:.2f} | {oc['mean_std_dev_deg'][2]:.2f} |")
        elif 'ori_cov' in s:
            report.append(f"| {source} | 0 | N/A | N/A | N/A |")

    report.append("\n## Key Findings\n")

    # Check GPS covariance quality
    if 'gps_odomenu' in stats and 'pos_cov' in stats['gps_odomenu']:
        pc = stats['gps_odomenu']['pos_cov']
        if pc['valid_count'] > 0:
            report.append(f"- GPS provides **dynamic position covariance** with range [{pc['min'][0]:.4f}, {pc['max'][0]:.4f}] m² for X")
            report.append(f"- Average GPS position uncertainty: ~{np.mean(pc['mean_std_dev']):.2f} m (1σ)")

    # Check LIO-SAM covariance
    if 'lio_odometry' in stats:
        if 'pos_cov' in stats['lio_odometry']:
            if stats['lio_odometry']['pos_cov']['all_zero']:
                report.append("- LIO-SAM **does NOT publish position covariance** in odometry message")
                report.append("  - The covariance is computed internally (poseCovariance) but not published")
                report.append("  - Consider modifying `publishOdometry()` to include covariance")
            else:
                pc = stats['lio_odometry']['pos_cov']
                report.append(f"- LIO-SAM publishes position covariance: mean std = {np.mean(pc['mean_std_dev']):.4f} m")

    report.append("\n## Recommendations\n")
    report.append("1. **GPS Covariance**: Use `useGpsSensorCovariance: true` to leverage dynamic GPS uncertainty")
    report.append("2. **LIO-SAM Covariance**: The internal `poseCovariance` from ISAM2 marginals is available but not published")
    report.append("3. To publish LIO-SAM covariance, modify `mapOptmization.cpp:publishOdometry()` to include `poseCovariance`")

    # Write report
    report_path = os.path.join(output_dir, 'gps_lio_covariance_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved to: {report_path}")

    return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze GPS and LIO-SAM covariances')
    parser.add_argument('bag_path', help='Path to ROS bag file')
    parser.add_argument('--output', '-o', default='/root/autodl-tmp/catkin_ws/src/LIO-SAM/output',
                       help='Output directory')
    parser.add_argument('--max-msgs', '-m', type=int, default=None,
                       help='Maximum messages per topic')
    args = parser.parse_args()

    # Extract data
    data = extract_covariance_from_bag(args.bag_path, args.max_msgs)

    # Compute statistics
    stats = compute_statistics(data)

    # Print statistics
    print_statistics(stats)

    # Visualize
    visualize_covariances(data, stats, args.output)

    # Generate report
    report = generate_report(data, stats, args.output)
    print("\n" + "="*80)
    print(report)


if __name__ == '__main__':
    main()
