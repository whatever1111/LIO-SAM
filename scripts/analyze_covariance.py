#!/usr/bin/env python3
"""
IMU/GPS Covariance Analyzer for LIO-SAM
Analyzes covariance data from:
- /fixposition/fpa/corrimu (FpaImu)
- /imu/data (sensor_msgs/Imu)
- /fixposition/fpa/odometry or /odometry/gps (GPS odometry)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    import rosbag
except ImportError:
    print("ERROR: rosbag not found. Run: source /opt/ros/noetic/setup.bash")
    sys.exit(1)


def analyze_bag_covariance(bag_path, max_messages=50000):
    """Extract and analyze covariance from bag file"""

    print(f"Opening bag file: {bag_path}")
    bag = rosbag.Bag(bag_path)

    results = {
        'fpa_corrimu': {
            'acc_cov': [], 'gyr_cov': [], 'ori_cov': [],
            'acc_data': [], 'gyr_data': [],
            'times': [], 'bias_comp': [], 'imu_status': []
        },
        'imu_data': {
            'acc_cov': [], 'gyr_cov': [], 'ori_cov': [],
            'acc_data': [], 'gyr_data': [], 'ori_data': [],
            'times': []
        },
        'fpa_odometry': {
            'pos_cov': [], 'ori_cov': [], 'full_cov': [],
            'pos_data': [], 'ori_data': [],
            'times': [], 'fusion_status': []
        },
        'fpa_odomenu': {
            'pos_cov': [], 'ori_cov': [], 'full_cov': [],
            'pos_data': [],
            'times': []
        }
    }

    # Topic mapping
    topic_handlers = {
        '/fixposition/fpa/corrimu': 'fpa_corrimu',
        '/imu/data': 'imu_data',
        '/fixposition/fpa/odometry': 'fpa_odometry',
        '/fixposition/fpa/odomenu': 'fpa_odomenu',
        '/odometry/gps': 'fpa_odomenu'  # Alternative GPS topic
    }

    msg_counts = defaultdict(int)

    print("Reading messages...")
    for topic, msg, t in bag.read_messages():
        if topic not in topic_handlers:
            continue

        handler = topic_handlers[topic]
        msg_counts[topic] += 1

        # Limit messages for faster processing
        if msg_counts[topic] > max_messages:
            continue

        timestamp = t.to_sec()

        if handler == 'fpa_corrimu':
            imu = msg.data
            results[handler]['times'].append(timestamp)
            results[handler]['acc_cov'].append(np.array(imu.linear_acceleration_covariance))
            results[handler]['gyr_cov'].append(np.array(imu.angular_velocity_covariance))
            results[handler]['ori_cov'].append(np.array(imu.orientation_covariance))
            results[handler]['acc_data'].append([imu.linear_acceleration.x,
                                                  imu.linear_acceleration.y,
                                                  imu.linear_acceleration.z])
            results[handler]['gyr_data'].append([imu.angular_velocity.x,
                                                  imu.angular_velocity.y,
                                                  imu.angular_velocity.z])
            results[handler]['bias_comp'].append(msg.bias_comp)
            results[handler]['imu_status'].append(msg.imu_status)

        elif handler == 'imu_data':
            results[handler]['times'].append(timestamp)
            results[handler]['acc_cov'].append(np.array(msg.linear_acceleration_covariance))
            results[handler]['gyr_cov'].append(np.array(msg.angular_velocity_covariance))
            results[handler]['ori_cov'].append(np.array(msg.orientation_covariance))
            results[handler]['acc_data'].append([msg.linear_acceleration.x,
                                                  msg.linear_acceleration.y,
                                                  msg.linear_acceleration.z])
            results[handler]['gyr_data'].append([msg.angular_velocity.x,
                                                  msg.angular_velocity.y,
                                                  msg.angular_velocity.z])
            results[handler]['ori_data'].append([msg.orientation.x, msg.orientation.y,
                                                  msg.orientation.z, msg.orientation.w])

        elif handler == 'fpa_odometry':
            results[handler]['times'].append(timestamp)
            # Handle both nav_msgs/Odometry and FpaOdometry
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                # nav_msgs/Odometry: pose.pose.position
                cov = np.array(msg.pose.covariance).reshape(6, 6)
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
            elif hasattr(msg, 'pose') and hasattr(msg.pose, 'covariance'):
                # PoseWithCovariance: pose.position
                cov = np.array(msg.pose.covariance).reshape(6, 6)
                pos = msg.pose.position if hasattr(msg.pose, 'position') else None
                ori = msg.pose.orientation if hasattr(msg.pose, 'orientation') else None
            else:
                continue
            results[handler]['full_cov'].append(cov)
            results[handler]['pos_cov'].append(cov[:3, :3])
            results[handler]['ori_cov'].append(cov[3:, 3:])
            if pos:
                results[handler]['pos_data'].append([pos.x, pos.y, pos.z])
            if ori:
                results[handler]['ori_data'].append([ori.x, ori.y, ori.z, ori.w])
            if hasattr(msg, 'fusion_status'):
                results[handler]['fusion_status'].append(msg.fusion_status)

        elif handler == 'fpa_odomenu':
            results[handler]['times'].append(timestamp)
            # Handle both nav_msgs/Odometry and FpaOdomenu
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
                # nav_msgs/Odometry
                cov = np.array(msg.pose.covariance).reshape(6, 6)
                pos = msg.pose.pose.position
            elif hasattr(msg, 'pose') and hasattr(msg.pose, 'covariance'):
                # PoseWithCovariance
                cov = np.array(msg.pose.covariance).reshape(6, 6)
                pos = msg.pose.position if hasattr(msg.pose, 'position') else None
            else:
                continue
            results[handler]['full_cov'].append(cov)
            results[handler]['pos_cov'].append(cov[:3, :3])
            results[handler]['ori_cov'].append(cov[3:, 3:])
            if pos:
                results[handler]['pos_data'].append([pos.x, pos.y, pos.z])

    bag.close()

    print(f"\nMessage counts:")
    for topic, count in msg_counts.items():
        print(f"  {topic}: {count}")

    return results


def analyze_covariance_statistics(results):
    """Compute statistics for each covariance type"""

    stats = {}

    for source, data in results.items():
        if not data['times']:
            continue

        stats[source] = {}
        duration = data['times'][-1] - data['times'][0]
        stats[source]['duration'] = duration
        stats[source]['count'] = len(data['times'])
        stats[source]['rate'] = len(data['times']) / duration if duration > 0 else 0

        # Analyze each covariance type
        for cov_name in ['acc_cov', 'gyr_cov', 'ori_cov', 'pos_cov']:
            if cov_name not in data or not data[cov_name]:
                continue

            covs = np.array(data[cov_name])

            # Handle both 9-element and 3x3 formats
            if covs.ndim == 2 and covs.shape[1] == 9:
                # 9-element array -> extract diagonal
                diag = covs[:, [0, 4, 8]]  # Elements 0, 4, 8 are diagonal
            elif covs.ndim == 3 and covs.shape[1:] == (3, 3):
                # 3x3 matrix -> extract diagonal
                diag = np.array([np.diag(c) for c in covs])
            else:
                continue

            cov_stats = {
                'shape': covs.shape,
                'all_zero': np.allclose(diag, 0),
                'all_same': np.allclose(diag, diag[0]) if len(diag) > 1 else True,
                'has_negative': np.any(diag < 0),
                'negative_count': np.sum(diag[:, 0] < 0),
            }

            # Check for valid (positive) covariances
            valid_mask = diag[:, 0] > 0
            cov_stats['valid_count'] = np.sum(valid_mask)
            cov_stats['valid_ratio'] = np.mean(valid_mask)

            if cov_stats['valid_count'] > 0:
                valid_diag = diag[valid_mask]
                cov_stats['mean'] = np.mean(valid_diag, axis=0)
                cov_stats['std'] = np.std(valid_diag, axis=0)
                cov_stats['min'] = np.min(valid_diag, axis=0)
                cov_stats['max'] = np.max(valid_diag, axis=0)
                cov_stats['range_ratio'] = cov_stats['max'] / (cov_stats['min'] + 1e-10)

            # Check if covariance is dynamic (changes over time)
            if len(diag) > 10:
                cov_stats['is_dynamic'] = not np.allclose(diag[::10], diag[0])
                cov_stats['variation_coef'] = np.std(diag, axis=0) / (np.mean(diag, axis=0) + 1e-10)

            stats[source][cov_name] = cov_stats

    return stats


def print_statistics_report(stats):
    """Print formatted statistics report"""

    print("\n" + "="*80)
    print("COVARIANCE ANALYSIS REPORT")
    print("="*80)

    for source, source_stats in stats.items():
        print(f"\n{'─'*40}")
        print(f"Source: {source}")
        print(f"{'─'*40}")
        print(f"  Messages: {source_stats['count']}")
        print(f"  Duration: {source_stats['duration']:.2f}s")
        print(f"  Rate: {source_stats['rate']:.1f} Hz")

        for cov_name in ['acc_cov', 'gyr_cov', 'ori_cov', 'pos_cov']:
            if cov_name not in source_stats:
                continue
            cov = source_stats[cov_name]

            print(f"\n  {cov_name}:")
            print(f"    All zero: {cov['all_zero']}")
            print(f"    All same value: {cov['all_same']}")
            print(f"    Has negative (invalid): {cov['has_negative']} ({cov['negative_count']} msgs)")
            print(f"    Valid count: {cov['valid_count']} ({cov['valid_ratio']*100:.1f}%)")

            if cov['valid_count'] > 0:
                print(f"    Mean (diag): [{cov['mean'][0]:.2e}, {cov['mean'][1]:.2e}, {cov['mean'][2]:.2e}]")
                print(f"    Std (diag):  [{cov['std'][0]:.2e}, {cov['std'][1]:.2e}, {cov['std'][2]:.2e}]")
                print(f"    Min (diag):  [{cov['min'][0]:.2e}, {cov['min'][1]:.2e}, {cov['min'][2]:.2e}]")
                print(f"    Max (diag):  [{cov['max'][0]:.2e}, {cov['max'][1]:.2e}, {cov['max'][2]:.2e}]")

                if 'is_dynamic' in cov:
                    print(f"    Is dynamic: {cov['is_dynamic']}")
                    if cov['is_dynamic']:
                        print(f"    Variation coef: [{cov['variation_coef'][0]:.2f}, {cov['variation_coef'][1]:.2f}, {cov['variation_coef'][2]:.2f}]")


def visualize_covariances(results, stats, output_dir):
    """Generate visualization plots"""

    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    fig.suptitle('Covariance Analysis', fontsize=14)

    plot_idx = 0

    # Row 0: FPA CORRIMU
    if results['fpa_corrimu']['times']:
        t = np.array(results['fpa_corrimu']['times'])
        t = t - t[0]

        # Acceleration covariance
        acc_cov = np.array(results['fpa_corrimu']['acc_cov'])
        if acc_cov.shape[1] == 9:
            acc_diag = acc_cov[:, [0, 4, 8]]
        ax = axes[0, 0]
        ax.plot(t, acc_diag, alpha=0.7)
        ax.set_title('FPA CORRIMU - Acc Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Variance')
        ax.legend(['X', 'Y', 'Z'], loc='upper right')
        ax.set_yscale('symlog', linthresh=1e-10)

        # Gyro covariance
        gyr_cov = np.array(results['fpa_corrimu']['gyr_cov'])
        if gyr_cov.shape[1] == 9:
            gyr_diag = gyr_cov[:, [0, 4, 8]]
        ax = axes[0, 1]
        ax.plot(t, gyr_diag, alpha=0.7)
        ax.set_title('FPA CORRIMU - Gyro Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.set_yscale('symlog', linthresh=1e-10)

        # IMU status
        ax = axes[0, 2]
        imu_status = np.array(results['fpa_corrimu']['imu_status'])
        bias_comp = np.array(results['fpa_corrimu']['bias_comp'])
        ax.plot(t, imu_status, 'b-', alpha=0.7, label='imu_status')
        ax.plot(t, bias_comp.astype(int), 'r-', alpha=0.7, label='bias_comp')
        ax.set_title('FPA CORRIMU - Status')
        ax.set_xlabel('Time (s)')
        ax.legend()

    # Row 1: /imu/data
    if results['imu_data']['times']:
        t = np.array(results['imu_data']['times'])
        t = t - t[0]

        # Acceleration covariance
        acc_cov = np.array(results['imu_data']['acc_cov'])
        if acc_cov.shape[1] == 9:
            acc_diag = acc_cov[:, [0, 4, 8]]
        ax = axes[1, 0]
        ax.plot(t, acc_diag, alpha=0.7)
        ax.set_title('/imu/data - Acc Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.set_yscale('symlog', linthresh=1e-10)

        # Gyro covariance
        gyr_cov = np.array(results['imu_data']['gyr_cov'])
        if gyr_cov.shape[1] == 9:
            gyr_diag = gyr_cov[:, [0, 4, 8]]
        ax = axes[1, 1]
        ax.plot(t, gyr_diag, alpha=0.7)
        ax.set_title('/imu/data - Gyro Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.set_yscale('symlog', linthresh=1e-10)

        # Orientation covariance
        ori_cov = np.array(results['imu_data']['ori_cov'])
        if ori_cov.shape[1] == 9:
            ori_diag = ori_cov[:, [0, 4, 8]]
        ax = axes[1, 2]
        ax.plot(t, ori_diag, alpha=0.7)
        ax.set_title('/imu/data - Orientation Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.set_yscale('symlog', linthresh=1e-10)

    # Row 2: FPA Odometry
    if results['fpa_odometry']['times']:
        t = np.array(results['fpa_odometry']['times'])
        t = t - t[0]

        # Position covariance
        pos_cov = np.array(results['fpa_odometry']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])
        ax = axes[2, 0]
        ax.semilogy(t, pos_diag, alpha=0.7)
        ax.set_title('FPA Odometry - Position Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.legend(['X', 'Y', 'Z'], loc='upper right')

        # Orientation covariance
        ori_cov = np.array(results['fpa_odometry']['ori_cov'])
        ori_diag = np.array([np.diag(c) for c in ori_cov])
        ax = axes[2, 1]
        ax.semilogy(t, ori_diag, alpha=0.7)
        ax.set_title('FPA Odometry - Orientation Cov (diag)')
        ax.set_xlabel('Time (s)')
        ax.legend(['Roll', 'Pitch', 'Yaw'], loc='upper right')

        # Cross-correlation (off-diagonal)
        ax = axes[2, 2]
        full_cov = np.array(results['fpa_odometry']['full_cov'])
        # Position-orientation correlation
        cross = np.array([c[0, 3] for c in full_cov])  # x-roll correlation
        ax.plot(t, cross, alpha=0.7)
        ax.set_title('FPA Odometry - Cross Correlation (x-roll)')
        ax.set_xlabel('Time (s)')

    # Row 3: FPA Odomenu (ENU)
    if results['fpa_odomenu']['times']:
        t = np.array(results['fpa_odomenu']['times'])
        t = t - t[0]

        # Position covariance
        pos_cov = np.array(results['fpa_odomenu']['pos_cov'])
        pos_diag = np.array([np.diag(c) for c in pos_cov])
        ax = axes[3, 0]
        ax.semilogy(t, pos_diag, alpha=0.7)
        ax.set_title('FPA Odomenu (ENU) - Position Cov')
        ax.set_xlabel('Time (s)')
        ax.legend(['E', 'N', 'U'], loc='upper right')

        # Orientation covariance
        ori_cov = np.array(results['fpa_odomenu']['ori_cov'])
        ori_diag = np.array([np.diag(c) for c in ori_cov])
        ax = axes[3, 1]
        ax.semilogy(t, ori_diag, alpha=0.7)
        ax.set_title('FPA Odomenu - Orientation Cov')
        ax.set_xlabel('Time (s)')

        # Covariance histogram
        ax = axes[3, 2]
        ax.hist(pos_diag[:, 0], bins=50, alpha=0.7, label='E')
        ax.hist(pos_diag[:, 1], bins=50, alpha=0.7, label='N')
        ax.set_title('Position Cov Distribution')
        ax.set_xlabel('Variance (m²)')
        ax.legend()

    plt.tight_layout()
    output_path = f'{output_dir}/covariance_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def generate_recommendations(stats, output_dir):
    """Generate recommendations for params.yaml based on covariance analysis"""

    report = []
    report.append("# Covariance Analysis Recommendations")
    report.append("")
    report.append("## Summary")
    report.append("")

    # Analyze FPA CORRIMU
    if 'fpa_corrimu' in stats:
        report.append("### FPA CORRIMU (/fixposition/fpa/corrimu)")
        s = stats['fpa_corrimu']

        if 'acc_cov' in s:
            acc = s['acc_cov']
            if acc['all_zero']:
                report.append("- **Accelerometer Covariance**: ALL ZERO - sensor does not provide covariance")
                report.append("  - Recommendation: Use fixed noise parameters in params.yaml")
            elif acc['valid_count'] > 0:
                report.append(f"- **Accelerometer Covariance**: Valid ({acc['valid_ratio']*100:.1f}%)")
                report.append(f"  - Mean: {acc['mean']}")
                report.append(f"  - Recommended imuAccNoise: {np.sqrt(np.mean(acc['mean'])):.6f}")

        if 'gyr_cov' in s:
            gyr = s['gyr_cov']
            if gyr['all_zero']:
                report.append("- **Gyroscope Covariance**: ALL ZERO - sensor does not provide covariance")
            elif gyr['valid_count'] > 0:
                report.append(f"- **Gyroscope Covariance**: Valid ({gyr['valid_ratio']*100:.1f}%)")
                report.append(f"  - Recommended imuGyrNoise: {np.sqrt(np.mean(gyr['mean'])):.6f}")

        report.append("")

    # Analyze /imu/data
    if 'imu_data' in stats:
        report.append("### Standard IMU (/imu/data)")
        s = stats['imu_data']

        for cov_name, label in [('acc_cov', 'Accelerometer'),
                                 ('gyr_cov', 'Gyroscope'),
                                 ('ori_cov', 'Orientation')]:
            if cov_name in s:
                cov = s[cov_name]
                if cov['all_zero']:
                    report.append(f"- **{label} Covariance**: ALL ZERO")
                elif cov['valid_count'] > 0:
                    report.append(f"- **{label} Covariance**: Valid, Dynamic={cov.get('is_dynamic', 'N/A')}")
                    report.append(f"  - Mean: [{cov['mean'][0]:.2e}, {cov['mean'][1]:.2e}, {cov['mean'][2]:.2e}]")
        report.append("")

    # Analyze GPS/Odometry
    for source in ['fpa_odometry', 'fpa_odomenu']:
        if source in stats:
            report.append(f"### {source}")
            s = stats[source]

            if 'pos_cov' in s:
                pos = s['pos_cov']
                if pos['valid_count'] > 0:
                    report.append(f"- **Position Covariance**: Dynamic={pos.get('is_dynamic', False)}")
                    report.append(f"  - Mean: [{pos['mean'][0]:.4f}, {pos['mean'][1]:.4f}, {pos['mean'][2]:.4f}] m²")
                    report.append(f"  - Range: [{pos['min'][0]:.4f}, {pos['max'][0]:.4f}] m² (X)")
                    if pos.get('is_dynamic', False):
                        report.append(f"  - **Recommendation**: USE sensor covariance (useGpsSensorCovariance: true)")
                    else:
                        report.append(f"  - Recommendation: Fixed covariance is acceptable")
            report.append("")

    # GTSAM Compatibility Analysis
    report.append("## GTSAM Dynamic Covariance Compatibility")
    report.append("")
    report.append("### Why Dynamic IMU Covariance is NOT Suitable for Current Code:")
    report.append("")
    report.append("1. **PreintegratedImuMeasurements Design**:")
    report.append("   - GTSAM's IMU preintegration uses FIXED noise parameters set at construction time")
    report.append("   - `PreintegrationParams` defines `accelerometerCovariance` and `gyroscopeCovariance`")
    report.append("   - These are continuous-time white noise parameters, NOT per-measurement covariances")
    report.append("")
    report.append("2. **Preintegration Theory**:")
    report.append("   - IMU preintegration accumulates measurements between keyframes")
    report.append("   - Uncertainty grows via covariance propagation: Σ(t) = F*Σ(t-1)*F' + Q")
    report.append("   - Q is process noise, assumed constant for Gaussian white noise model")
    report.append("   - Changing Q per-measurement breaks the preintegration theory")
    report.append("")
    report.append("3. **Current Code Implementation (imuPreintegration.cpp:246-248)**:")
    report.append("   ```cpp")
    report.append("   p->accelerometerCovariance = Matrix33::Identity() * pow(imuAccNoise, 2);")
    report.append("   p->gyroscopeCovariance = Matrix33::Identity() * pow(imuGyrNoise, 2);")
    report.append("   ```")
    report.append("   - Noise is set ONCE at initialization")
    report.append("   - Cannot be changed per-measurement without recreating the integrator")
    report.append("")
    report.append("4. **Alternatives for Using Dynamic Covariance**:")
    report.append("   - **Option A**: Use average covariance to tune imuAccNoise/imuGyrNoise")
    report.append("   - **Option B**: Reject high-covariance IMU measurements (outlier rejection)")
    report.append("   - **Option C**: Use CombinedImuFactor with time-varying bias model")
    report.append("   - **Option D**: Scale pose correction noise based on IMU quality")
    report.append("")

    # Write report
    report_path = f'{output_dir}/covariance_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved to: {report_path}")

    return '\n'.join(report)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze IMU/GPS covariance from bag file')
    parser.add_argument('bag_path', help='Path to ROS bag file')
    parser.add_argument('--output', '-o', default='/root/autodl-tmp/catkin_ws/src/LIO-SAM/output',
                        help='Output directory for plots and reports')
    parser.add_argument('--max-msgs', '-m', type=int, default=50000,
                        help='Maximum messages to process per topic')
    args = parser.parse_args()

    # Analyze
    results = analyze_bag_covariance(args.bag_path, args.max_msgs)
    stats = analyze_covariance_statistics(results)

    # Print report
    print_statistics_report(stats)

    # Visualize
    visualize_covariances(results, stats, args.output)

    # Generate recommendations
    report = generate_recommendations(stats, args.output)
    print("\n" + "="*80)
    print(report)


if __name__ == '__main__':
    main()
