#!/usr/bin/env python3
"""
Plot all CORRIMU data gaps with bias_comp, imu_status fields,
and additional status from FpaOdometry (fusion_status, imu_bias_status, gnss_status).
Each gap generates a separate image showing data around the gap.
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

BAG_FILE = "/root/autodl-tmp/info2.bag"
OUTPUT_DIR = "/root/autodl-tmp/catkin_ws/src/LIO-SAM/output/gaps"

# IMU_STATUS constants from FpaConsts.msg (for FpaImu.imu_status)
IMU_STATUS_NAMES = {
    -1: "UNSPEC",
    0: "NOT_CONV",
    1: "WARM",
    2: "ROUGH",
    3: "FINE"
}

# IMU_STATUS_LEGACY constants (for FpaOdometry.imu_bias_status)
IMU_STATUS_LEGACY_NAMES = {
    -1: "UNSPEC",
    0: "NOT_CONV",
    1: "CONVERGED"
}

# FUSION_STATUS_LEGACY constants
FUSION_STATUS_LEGACY_NAMES = {
    -1: "UNSPEC",
    0: "NOT_INIT",
    1: "LOCAL_INIT",
    2: "GLOBAL_INIT"
}

# GNSS_FIX constants
GNSS_FIX_NAMES = {
    -1: "UNSPEC",
    0: "UNKNOWN",
    1: "NOFIX",
    2: "DRONLY",
    3: "TIME",
    4: "2D",
    5: "3D",
    6: "3D+DR",
    7: "RTK_FLOAT",
    8: "RTK_FIXED",
    9: "RTK_FLOAT+DR",
    10: "RTK_FIXED+DR"
}


def plot_gap(corrimu_times, corrimu_accs, corrimu_gyros, corrimu_bias_comp, corrimu_imu_status,
             imu_times, imu_accs, imu_gyros,
             odom_times, odom_fusion_status, odom_imu_bias_status, odom_gnss1_status, odom_gnss2_status,
             gap_start, gap_end, gap_idx, output_dir, context_before=5, context_after=5):
    """Plot a single gap with context before and after."""

    t_start = gap_start - context_before
    t_end = gap_end + context_after

    # Filter data for this window
    corrimu_mask = (corrimu_times >= t_start) & (corrimu_times <= t_end)
    imu_mask = (imu_times >= t_start) & (imu_times <= t_end)
    odom_mask = (odom_times >= t_start) & (odom_times <= t_end)

    ct = corrimu_times[corrimu_mask]
    it = imu_times[imu_mask]
    ot = odom_times[odom_mask]

    if len(ct) == 0 and len(it) == 0:
        print(f"  Warning: No data in window [{t_start:.1f}, {t_end:.1f}]s, skipping")
        return

    # Calculate magnitudes
    corrimu_acc_mag = np.linalg.norm(corrimu_accs, axis=1)
    imu_acc_mag = np.linalg.norm(imu_accs, axis=1)
    corrimu_gyro_mag = np.linalg.norm(corrimu_gyros, axis=1)
    imu_gyro_mag = np.linalg.norm(imu_gyros, axis=1)

    gap_duration = gap_end - gap_start

    fig, axes = plt.subplots(6, 2, figsize=(20, 24))
    fig.suptitle(f'CORRIMU Gap #{gap_idx}: t={gap_start:.3f}s to {gap_end:.3f}s (duration={gap_duration*1000:.1f}ms)',
                 fontsize=14, fontweight='bold')

    # Row 0: CORRIMU bias_comp and imu_status
    ax = axes[0, 0]
    if len(ct) > 0:
        ax.scatter(ct, corrimu_bias_comp[corrimu_mask], c='purple', s=10, alpha=0.7, label='bias_comp')
    ax.set_ylabel('bias_comp')
    ax.set_title('CORRIMU bias_comp (True=bias compensated)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['False', 'True'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red', label=f'Gap ({gap_duration*1000:.0f}ms)')
    ax.legend()

    ax = axes[0, 1]
    if len(ct) > 0:
        ax.scatter(ct, corrimu_imu_status[corrimu_mask], c='orange', s=10, alpha=0.7, label='imu_status')
    ax.set_ylabel('imu_status')
    ax.set_title('CORRIMU imu_status')
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.set_yticklabels(['UNSPEC', 'NOT_CONV', 'WARM', 'ROUGH', 'FINE'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red', label=f'Gap ({gap_duration*1000:.0f}ms)')
    ax.legend()

    # Row 1: FpaOdometry status (fusion_status, imu_bias_status)
    ax = axes[1, 0]
    if len(ot) > 0:
        ax.scatter(ot, odom_fusion_status[odom_mask], c='blue', s=15, alpha=0.7, label='fusion_status')
    ax.set_ylabel('fusion_status')
    ax.set_title('FpaOdometry fusion_status')
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(['UNSPEC', 'NOT_INIT', 'LOCAL', 'GLOBAL'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')
    ax.legend()

    ax = axes[1, 1]
    if len(ot) > 0:
        ax.scatter(ot, odom_imu_bias_status[odom_mask], c='green', s=15, alpha=0.7, label='imu_bias_status')
    ax.set_ylabel('imu_bias_status')
    ax.set_title('FpaOdometry imu_bias_status')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['UNSPEC', 'NOT_CONV', 'CONVERGED'])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')
    ax.legend()

    # Row 2: GNSS status
    ax = axes[2, 0]
    if len(ot) > 0:
        ax.scatter(ot, odom_gnss1_status[odom_mask], c='cyan', s=15, alpha=0.7, label='gnss1_status')
    ax.set_ylabel('gnss1_status')
    ax.set_title('FpaOdometry gnss1_status')
    ax.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticklabels(['UNSPEC', 'UNK', 'NOFIX', 'DR', 'TIME', '2D', '3D', '3D+DR', 'RTK_F', 'RTK', 'RTK_F+DR', 'RTK+DR'], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')
    ax.legend()

    ax = axes[2, 1]
    if len(ot) > 0:
        ax.scatter(ot, odom_gnss2_status[odom_mask], c='magenta', s=15, alpha=0.7, label='gnss2_status')
    ax.set_ylabel('gnss2_status')
    ax.set_title('FpaOdometry gnss2_status')
    ax.set_yticks([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticklabels(['UNSPEC', 'UNK', 'NOFIX', 'DR', 'TIME', '2D', '3D', '3D+DR', 'RTK_F', 'RTK', 'RTK_F+DR', 'RTK+DR'], fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')
    ax.legend()

    # Row 3: Acc Magnitude, Gyro Magnitude
    ax = axes[3, 0]
    if len(ct) > 0:
        ax.plot(ct, corrimu_acc_mag[corrimu_mask], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    if len(it) > 0:
        ax.plot(it, imu_acc_mag[imu_mask], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.axhline(y=9.81, color='g', linestyle='--', alpha=0.5, label='gravity')
    ax.set_ylabel('Acc Magnitude (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    ax = axes[3, 1]
    if len(ct) > 0:
        ax.plot(ct, corrimu_gyro_mag[corrimu_mask], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    if len(it) > 0:
        ax.plot(it, imu_gyro_mag[imu_mask], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Gyro Magnitude (rad/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    # Row 4: Acc X, Y, Z
    ax = axes[4, 0]
    if len(ct) > 0:
        ax.plot(ct, corrimu_accs[corrimu_mask, 0], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU X')
        ax.plot(ct, corrimu_accs[corrimu_mask, 1], 'g-', alpha=0.7, linewidth=0.8, label='CORRIMU Y')
        ax.plot(ct, corrimu_accs[corrimu_mask, 2], 'b-', alpha=0.7, linewidth=0.8, label='CORRIMU Z')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('CORRIMU Acceleration X/Y/Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    ax = axes[4, 1]
    if len(it) > 0:
        ax.plot(it, imu_accs[imu_mask, 0], 'r-', alpha=0.7, linewidth=0.8, label='/imu/data X')
        ax.plot(it, imu_accs[imu_mask, 1], 'g-', alpha=0.7, linewidth=0.8, label='/imu/data Y')
        ax.plot(it, imu_accs[imu_mask, 2], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data Z')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('/imu/data Acceleration X/Y/Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    # Row 5: Gyro X, Y, Z
    ax = axes[5, 0]
    if len(ct) > 0:
        ax.plot(ct, corrimu_gyros[corrimu_mask, 0], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU X')
        ax.plot(ct, corrimu_gyros[corrimu_mask, 1], 'g-', alpha=0.7, linewidth=0.8, label='CORRIMU Y')
        ax.plot(ct, corrimu_gyros[corrimu_mask, 2], 'b-', alpha=0.7, linewidth=0.8, label='CORRIMU Z')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('CORRIMU Angular Velocity X/Y/Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    ax = axes[5, 1]
    if len(it) > 0:
        ax.plot(it, imu_gyros[imu_mask, 0], 'r-', alpha=0.7, linewidth=0.8, label='/imu/data X')
        ax.plot(it, imu_gyros[imu_mask, 1], 'g-', alpha=0.7, linewidth=0.8, label='/imu/data Y')
        ax.plot(it, imu_gyros[imu_mask, 2], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data Z')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('/imu/data Angular Velocity X/Y/Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])
    ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    plt.tight_layout()

    filename = f"gap_{gap_idx:03d}_t{gap_start:.1f}s_{gap_duration*1000:.0f}ms.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def main():
    parser = argparse.ArgumentParser(description='Plot CORRIMU data gaps with status information')
    parser.add_argument('--gap-threshold', type=float, default=1.0,
                        help='Gap threshold in seconds (default: 1.0)')
    parser.add_argument('--context-before', type=float, default=5.0,
                        help='Context before gap in seconds (default: 5.0)')
    parser.add_argument('--context-after', type=float, default=5.0,
                        help='Context after gap in seconds (default: 5.0)')
    parser.add_argument('--bag', type=str, default=BAG_FILE,
                        help=f'Bag file path (default: {BAG_FILE})')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    bag = rosbag.Bag(args.bag)

    corrimu_data = []
    imu_data = []
    odom_data = []

    print(f"Reading bag file: {args.bag}")
    print(f"Gap threshold: {args.gap_threshold}s")

    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/corrimu', '/imu/data', '/fixposition/fpa/odometry']):
        if topic == '/fixposition/fpa/corrimu':
            ts = msg.data.header.stamp.to_sec()
            acc = np.array([msg.data.linear_acceleration.x,
                           msg.data.linear_acceleration.y,
                           msg.data.linear_acceleration.z])
            gyro = np.array([msg.data.angular_velocity.x,
                            msg.data.angular_velocity.y,
                            msg.data.angular_velocity.z])
            bias_comp = msg.bias_comp
            imu_status = msg.imu_status
            corrimu_data.append((ts, acc, gyro, bias_comp, imu_status))

        elif topic == '/imu/data':
            ts = msg.header.stamp.to_sec()
            acc = np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z])
            gyro = np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
            imu_data.append((ts, acc, gyro))

        elif topic == '/fixposition/fpa/odometry':
            ts = msg.header.stamp.to_sec()
            fusion_status = msg.fusion_status
            imu_bias_status = msg.imu_bias_status
            gnss1_status = msg.gnss1_status
            gnss2_status = msg.gnss2_status
            odom_data.append((ts, fusion_status, imu_bias_status, gnss1_status, gnss2_status))

    bag.close()

    print(f"CORRIMU: {len(corrimu_data)} messages")
    print(f"/imu/data: {len(imu_data)} messages")
    print(f"FpaOdometry: {len(odom_data)} messages")

    # Convert to arrays
    corrimu_times = np.array([d[0] for d in corrimu_data])
    corrimu_accs = np.array([d[1] for d in corrimu_data])
    corrimu_gyros = np.array([d[2] for d in corrimu_data])
    corrimu_bias_comp = np.array([d[3] for d in corrimu_data])
    corrimu_imu_status = np.array([d[4] for d in corrimu_data])

    imu_times = np.array([d[0] for d in imu_data])
    imu_accs = np.array([d[1] for d in imu_data])
    imu_gyros = np.array([d[2] for d in imu_data])

    odom_times = np.array([d[0] for d in odom_data]) if odom_data else np.array([])
    odom_fusion_status = np.array([d[1] for d in odom_data]) if odom_data else np.array([])
    odom_imu_bias_status = np.array([d[2] for d in odom_data]) if odom_data else np.array([])
    odom_gnss1_status = np.array([d[3] for d in odom_data]) if odom_data else np.array([])
    odom_gnss2_status = np.array([d[4] for d in odom_data]) if odom_data else np.array([])

    # Normalize times to start from 0
    t0 = min(corrimu_times[0], imu_times[0])
    corrimu_times -= t0
    imu_times -= t0
    if len(odom_times) > 0:
        odom_times -= t0

    # Print status statistics
    print("\n--- CORRIMU Status Statistics ---")
    unique_bias, counts_bias = np.unique(corrimu_bias_comp, return_counts=True)
    for val, cnt in zip(unique_bias, counts_bias):
        print(f"  bias_comp={val}: {cnt} ({100*cnt/len(corrimu_bias_comp):.2f}%)")

    unique_status, counts_status = np.unique(corrimu_imu_status, return_counts=True)
    for val, cnt in zip(unique_status, counts_status):
        name = IMU_STATUS_NAMES.get(val, f"UNKNOWN({val})")
        print(f"  imu_status={val} ({name}): {cnt} ({100*cnt/len(corrimu_imu_status):.2f}%)")

    if len(odom_data) > 0:
        print("\n--- FpaOdometry Status Statistics ---")
        unique_fusion, counts_fusion = np.unique(odom_fusion_status, return_counts=True)
        for val, cnt in zip(unique_fusion, counts_fusion):
            name = FUSION_STATUS_LEGACY_NAMES.get(val, f"UNKNOWN({val})")
            print(f"  fusion_status={val} ({name}): {cnt} ({100*cnt/len(odom_fusion_status):.2f}%)")

        unique_imu_bias, counts_imu_bias = np.unique(odom_imu_bias_status, return_counts=True)
        for val, cnt in zip(unique_imu_bias, counts_imu_bias):
            name = IMU_STATUS_LEGACY_NAMES.get(val, f"UNKNOWN({val})")
            print(f"  imu_bias_status={val} ({name}): {cnt} ({100*cnt/len(odom_imu_bias_status):.2f}%)")

        unique_gnss1, counts_gnss1 = np.unique(odom_gnss1_status, return_counts=True)
        for val, cnt in zip(unique_gnss1, counts_gnss1):
            name = GNSS_FIX_NAMES.get(val, f"UNKNOWN({val})")
            print(f"  gnss1_status={val} ({name}): {cnt} ({100*cnt/len(odom_gnss1_status):.2f}%)")

    # Find gaps
    corrimu_dt = np.diff(corrimu_times)
    gap_indices = np.where(corrimu_dt > args.gap_threshold)[0]

    print(f"\nFound {len(gap_indices)} gaps > {args.gap_threshold}s")

    if len(gap_indices) == 0:
        print("No gaps found. Try lowering --gap-threshold")
        return

    # Print gap summary
    print("\n--- Gap Summary ---")
    for i, idx in enumerate(gap_indices):
        gap_start = corrimu_times[idx]
        gap_end = corrimu_times[idx + 1]
        gap_duration = corrimu_dt[idx]
        print(f"  Gap #{i+1}: t={gap_start:.3f}s -> {gap_end:.3f}s (duration={gap_duration*1000:.1f}ms)")

    # Generate plots for each gap
    print(f"\nGenerating plots to {args.output}/")

    for i, idx in enumerate(gap_indices):
        gap_start = corrimu_times[idx]
        gap_end = corrimu_times[idx + 1]
        gap_duration = corrimu_dt[idx]

        print(f"  Plotting gap #{i+1}: t={gap_start:.3f}s ({gap_duration*1000:.1f}ms)...", end=" ")

        filepath = plot_gap(
            corrimu_times, corrimu_accs, corrimu_gyros, corrimu_bias_comp, corrimu_imu_status,
            imu_times, imu_accs, imu_gyros,
            odom_times, odom_fusion_status, odom_imu_bias_status, odom_gnss1_status, odom_gnss2_status,
            gap_start, gap_end, i+1, args.output,
            context_before=args.context_before,
            context_after=args.context_after
        )

        if filepath:
            print(f"saved")
        else:
            print(f"skipped")

    # Generate summary plot showing all gaps on timeline
    fig, axes = plt.subplots(5, 1, figsize=(20, 16))
    fig.suptitle(f'CORRIMU Data Gaps Overview (threshold > {args.gap_threshold}s)', fontsize=14, fontweight='bold')

    # Plot 1: Time intervals
    ax = axes[0]
    ax.semilogy(corrimu_times[:-1], corrimu_dt * 1000, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(y=args.gap_threshold * 1000, color='r', linestyle='--', label=f'Threshold ({args.gap_threshold}s)')
    ax.axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Normal (5ms)')
    ax.set_ylabel('Time Interval (ms, log scale)')
    ax.set_title('CORRIMU Time Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for idx in gap_indices:
        ax.axvline(x=corrimu_times[idx], color='red', alpha=0.5, linewidth=2)

    # Plot 2: Acceleration magnitude
    corrimu_acc_mag = np.linalg.norm(corrimu_accs, axis=1)
    ax = axes[1]
    ax.plot(corrimu_times, corrimu_acc_mag, 'r-', linewidth=0.3, alpha=0.7, label='CORRIMU')
    ax.axhline(y=9.81, color='g', linestyle='--', alpha=0.5)
    ax.set_ylabel('Acc Magnitude (m/s²)')
    ax.set_title('Acceleration Magnitude with Gap Locations')
    ax.grid(True, alpha=0.3)
    for i, idx in enumerate(gap_indices):
        gap_start = corrimu_times[idx]
        gap_end = corrimu_times[idx + 1]
        ax.axvspan(gap_start, gap_end, alpha=0.3, color='red',
                   label=f'Gap' if i == 0 else None)
    ax.legend()

    # Plot 3: CORRIMU imu_status
    ax = axes[2]
    ax.scatter(corrimu_times, corrimu_imu_status, c='orange', s=1, alpha=0.5)
    ax.set_ylabel('imu_status')
    ax.set_title('CORRIMU imu_status')
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.set_yticklabels(['UNSPEC', 'NOT_CONV', 'WARM', 'ROUGH', 'FINE'])
    ax.grid(True, alpha=0.3)
    for idx in gap_indices:
        gap_start = corrimu_times[idx]
        gap_end = corrimu_times[idx + 1]
        ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    # Plot 4: FpaOdometry fusion_status
    ax = axes[3]
    if len(odom_times) > 0:
        ax.scatter(odom_times, odom_fusion_status, c='blue', s=2, alpha=0.5)
    ax.set_ylabel('fusion_status')
    ax.set_title('FpaOdometry fusion_status')
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_yticklabels(['UNSPEC', 'NOT_INIT', 'LOCAL', 'GLOBAL'])
    ax.grid(True, alpha=0.3)
    for idx in gap_indices:
        gap_start = corrimu_times[idx]
        gap_end = corrimu_times[idx + 1]
        ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    # Plot 5: GNSS status
    ax = axes[4]
    if len(odom_times) > 0:
        ax.scatter(odom_times, odom_gnss1_status, c='cyan', s=2, alpha=0.5, label='GNSS1')
        ax.scatter(odom_times, odom_gnss2_status, c='magenta', s=2, alpha=0.3, label='GNSS2')
    ax.set_ylabel('gnss_status')
    ax.set_xlabel('Time (s)')
    ax.set_title('FpaOdometry GNSS Status')
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.grid(True, alpha=0.3)
    ax.legend()
    for idx in gap_indices:
        gap_start = corrimu_times[idx]
        gap_end = corrimu_times[idx + 1]
        ax.axvspan(gap_start, gap_end, alpha=0.3, color='red')

    plt.tight_layout()
    summary_path = os.path.join(args.output, "gaps_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary: {summary_path}")

    print(f"\nDone! Generated {len(gap_indices)} gap images + 1 summary image")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
