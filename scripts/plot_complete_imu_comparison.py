#!/usr/bin/env python3
"""
Complete comparison of CORRIMU vs /imu/data
Including all acceleration (X, Y, Z, magnitude) and angular velocity (X, Y, Z)
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt

BAG_FILE = "/root/autodl-tmp/info.bag"
OUTPUT_DIR = "/root/autodl-tmp/catkin_ws/src/LIO-SAM/output"

def main():
    bag = rosbag.Bag(BAG_FILE)

    corrimu_data = []
    imu_data = []

    print("Reading bag file...")

    for topic, msg, t in bag.read_messages(topics=['/fixposition/fpa/corrimu', '/imu/data']):
        if topic == '/fixposition/fpa/corrimu':
            ts = msg.data.header.stamp.to_sec()
            acc = np.array([msg.data.linear_acceleration.x,
                           msg.data.linear_acceleration.y,
                           msg.data.linear_acceleration.z])
            gyro = np.array([msg.data.angular_velocity.x,
                            msg.data.angular_velocity.y,
                            msg.data.angular_velocity.z])
            corrimu_data.append((ts, acc, gyro))

        elif topic == '/imu/data':
            ts = msg.header.stamp.to_sec()
            acc = np.array([msg.linear_acceleration.x,
                           msg.linear_acceleration.y,
                           msg.linear_acceleration.z])
            gyro = np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
            imu_data.append((ts, acc, gyro))

    bag.close()

    print(f"CORRIMU: {len(corrimu_data)} messages")
    print(f"/imu/data: {len(imu_data)} messages")

    # Convert to arrays
    corrimu_times = np.array([d[0] for d in corrimu_data])
    corrimu_accs = np.array([d[1] for d in corrimu_data])
    corrimu_gyros = np.array([d[2] for d in corrimu_data])

    imu_times = np.array([d[0] for d in imu_data])
    imu_accs = np.array([d[1] for d in imu_data])
    imu_gyros = np.array([d[2] for d in imu_data])

    # Normalize times to start from 0
    t0 = min(corrimu_times[0], imu_times[0])
    corrimu_times -= t0
    imu_times -= t0

    # Calculate magnitudes
    corrimu_acc_mag = np.linalg.norm(corrimu_accs, axis=1)
    imu_acc_mag = np.linalg.norm(imu_accs, axis=1)
    corrimu_gyro_mag = np.linalg.norm(corrimu_gyros, axis=1)
    imu_gyro_mag = np.linalg.norm(imu_gyros, axis=1)

    # ==================== FULL TIME SERIES ====================
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle('CORRIMU vs /imu/data - Complete Comparison (Full Time Series)', fontsize=14, fontweight='bold')

    # Left column: Acceleration
    # Row 0: Acc Magnitude
    ax = axes[0, 0]
    ax.plot(corrimu_times, corrimu_acc_mag, 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_acc_mag, 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.axhline(y=9.81, color='g', linestyle='--', alpha=0.5, label='gravity (9.81)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='threshold (30)')
    ax.set_ylabel('Acc Magnitude (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 80])

    # Row 1: Acc X
    ax = axes[1, 0]
    ax.plot(corrimu_times, corrimu_accs[:, 0], 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_accs[:, 0], 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.set_ylabel('Acc X (m/s²)')
    ax.set_title('Acceleration X')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2: Acc Y
    ax = axes[2, 0]
    ax.plot(corrimu_times, corrimu_accs[:, 1], 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_accs[:, 1], 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.set_ylabel('Acc Y (m/s²)')
    ax.set_title('Acceleration Y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 3: Acc Z
    ax = axes[3, 0]
    ax.plot(corrimu_times, corrimu_accs[:, 2], 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_accs[:, 2], 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.axhline(y=9.81, color='g', linestyle='--', alpha=0.5, label='gravity')
    ax.set_ylabel('Acc Z (m/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Right column: Angular Velocity
    # Row 0: Gyro Magnitude
    ax = axes[0, 1]
    ax.plot(corrimu_times, corrimu_gyro_mag, 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_gyro_mag, 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.set_ylabel('Gyro Magnitude (rad/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 1: Gyro X
    ax = axes[1, 1]
    ax.plot(corrimu_times, corrimu_gyros[:, 0], 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_gyros[:, 0], 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.set_ylabel('Gyro X (rad/s)')
    ax.set_title('Angular Velocity X')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2: Gyro Y
    ax = axes[2, 1]
    ax.plot(corrimu_times, corrimu_gyros[:, 1], 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_gyros[:, 1], 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.set_ylabel('Gyro Y (rad/s)')
    ax.set_title('Angular Velocity Y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 3: Gyro Z
    ax = axes[3, 1]
    ax.plot(corrimu_times, corrimu_gyros[:, 2], 'r-', alpha=0.7, linewidth=0.5, label='CORRIMU')
    ax.plot(imu_times, imu_gyros[:, 2], 'b-', alpha=0.7, linewidth=0.5, label='/imu/data')
    ax.set_ylabel('Gyro Z (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Angular Velocity Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/imu_comparison_full.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/imu_comparison_full.png")
    plt.close()

    # ==================== ZOOMED VIEW (40-50s divergence period) ====================
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle('CORRIMU vs /imu/data - Zoomed View (t=40-50s, Divergence Period)', fontsize=14, fontweight='bold')

    t_start, t_end = 40, 50

    # Filter data for zoomed view
    corrimu_mask = (corrimu_times >= t_start) & (corrimu_times <= t_end)
    imu_mask = (imu_times >= t_start) & (imu_times <= t_end)

    # Left column: Acceleration
    ax = axes[0, 0]
    ax.plot(corrimu_times[corrimu_mask], corrimu_acc_mag[corrimu_mask], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_acc_mag[imu_mask], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.axhline(y=9.81, color='g', linestyle='--', alpha=0.5, label='gravity')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='threshold')
    ax.set_ylabel('Acc Magnitude (m/s²)')
    ax.set_title('Acceleration Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    ax = axes[1, 0]
    ax.plot(corrimu_times[corrimu_mask], corrimu_accs[corrimu_mask, 0], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_accs[imu_mask, 0], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Acc X (m/s²)')
    ax.set_title('Acceleration X')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    ax = axes[2, 0]
    ax.plot(corrimu_times[corrimu_mask], corrimu_accs[corrimu_mask, 1], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_accs[imu_mask, 1], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Acc Y (m/s²)')
    ax.set_title('Acceleration Y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    ax = axes[3, 0]
    ax.plot(corrimu_times[corrimu_mask], corrimu_accs[corrimu_mask, 2], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_accs[imu_mask, 2], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.axhline(y=9.81, color='g', linestyle='--', alpha=0.5, label='gravity')
    ax.set_ylabel('Acc Z (m/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    # Right column: Angular Velocity
    ax = axes[0, 1]
    ax.plot(corrimu_times[corrimu_mask], corrimu_gyro_mag[corrimu_mask], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_gyro_mag[imu_mask], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Gyro Magnitude (rad/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    ax = axes[1, 1]
    ax.plot(corrimu_times[corrimu_mask], corrimu_gyros[corrimu_mask, 0], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_gyros[imu_mask, 0], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Gyro X (rad/s)')
    ax.set_title('Angular Velocity X')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    ax = axes[2, 1]
    ax.plot(corrimu_times[corrimu_mask], corrimu_gyros[corrimu_mask, 1], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_gyros[imu_mask, 1], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Gyro Y (rad/s)')
    ax.set_title('Angular Velocity Y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    ax = axes[3, 1]
    ax.plot(corrimu_times[corrimu_mask], corrimu_gyros[corrimu_mask, 2], 'r-', alpha=0.7, linewidth=0.8, label='CORRIMU')
    ax.plot(imu_times[imu_mask], imu_gyros[imu_mask, 2], 'b-', alpha=0.7, linewidth=0.8, label='/imu/data')
    ax.set_ylabel('Gyro Z (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Angular Velocity Z')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([t_start, t_end])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/imu_comparison_zoomed.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/imu_comparison_zoomed.png")
    plt.close()

    # ==================== DIFFERENCE PLOT ====================
    # Interpolate /imu/data to CORRIMU timestamps for direct comparison
    from scipy.interpolate import interp1d

    imu_acc_interp = []
    imu_gyro_interp = []
    for i in range(3):
        f_acc = interp1d(imu_times, imu_accs[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        f_gyro = interp1d(imu_times, imu_gyros[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        imu_acc_interp.append(f_acc(corrimu_times))
        imu_gyro_interp.append(f_gyro(corrimu_times))

    imu_acc_interp = np.array(imu_acc_interp).T
    imu_gyro_interp = np.array(imu_gyro_interp).T

    acc_diff = corrimu_accs - imu_acc_interp
    gyro_diff = corrimu_gyros - imu_gyro_interp

    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle('CORRIMU - /imu/data Difference (Full Time Series)', fontsize=14, fontweight='bold')

    # Acc differences
    ax = axes[0, 0]
    acc_diff_mag = np.linalg.norm(acc_diff, axis=1)
    ax.plot(corrimu_times, acc_diff_mag, 'purple', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('|Acc Diff| (m/s²)')
    ax.set_title('Acceleration Difference Magnitude')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(corrimu_times, acc_diff[:, 0], 'r-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Acc X Diff (m/s²)')
    ax.set_title('Acceleration X Difference')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(corrimu_times, acc_diff[:, 1], 'g-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Acc Y Diff (m/s²)')
    ax.set_title('Acceleration Y Difference')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(corrimu_times, acc_diff[:, 2], 'b-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Acc Z Diff (m/s²)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration Z Difference')
    ax.grid(True, alpha=0.3)

    # Gyro differences
    ax = axes[0, 1]
    gyro_diff_mag = np.linalg.norm(gyro_diff, axis=1)
    ax.plot(corrimu_times, gyro_diff_mag, 'purple', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('|Gyro Diff| (rad/s)')
    ax.set_title('Angular Velocity Difference Magnitude')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(corrimu_times, gyro_diff[:, 0], 'r-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Gyro X Diff (rad/s)')
    ax.set_title('Angular Velocity X Difference')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(corrimu_times, gyro_diff[:, 1], 'g-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Gyro Y Diff (rad/s)')
    ax.set_title('Angular Velocity Y Difference')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(corrimu_times, gyro_diff[:, 2], 'b-', alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Gyro Z Diff (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Angular Velocity Z Difference')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/imu_comparison_diff.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}/imu_comparison_diff.png")
    plt.close()

    # ==================== STATISTICS ====================
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)

    print("\n--- Acceleration ---")
    print(f"CORRIMU:")
    print(f"  X: mean={np.mean(corrimu_accs[:,0]):+.4f}, std={np.std(corrimu_accs[:,0]):.4f}, max={np.max(np.abs(corrimu_accs[:,0])):.4f}")
    print(f"  Y: mean={np.mean(corrimu_accs[:,1]):+.4f}, std={np.std(corrimu_accs[:,1]):.4f}, max={np.max(np.abs(corrimu_accs[:,1])):.4f}")
    print(f"  Z: mean={np.mean(corrimu_accs[:,2]):+.4f}, std={np.std(corrimu_accs[:,2]):.4f}, max={np.max(np.abs(corrimu_accs[:,2])):.4f}")
    print(f"  Magnitude: mean={np.mean(corrimu_acc_mag):.4f}, std={np.std(corrimu_acc_mag):.4f}, max={np.max(corrimu_acc_mag):.4f}")

    print(f"\n/imu/data:")
    print(f"  X: mean={np.mean(imu_accs[:,0]):+.4f}, std={np.std(imu_accs[:,0]):.4f}, max={np.max(np.abs(imu_accs[:,0])):.4f}")
    print(f"  Y: mean={np.mean(imu_accs[:,1]):+.4f}, std={np.std(imu_accs[:,1]):.4f}, max={np.max(np.abs(imu_accs[:,1])):.4f}")
    print(f"  Z: mean={np.mean(imu_accs[:,2]):+.4f}, std={np.std(imu_accs[:,2]):.4f}, max={np.max(np.abs(imu_accs[:,2])):.4f}")
    print(f"  Magnitude: mean={np.mean(imu_acc_mag):.4f}, std={np.std(imu_acc_mag):.4f}, max={np.max(imu_acc_mag):.4f}")

    print("\n--- Angular Velocity ---")
    print(f"CORRIMU:")
    print(f"  X: mean={np.mean(corrimu_gyros[:,0]):+.6f}, std={np.std(corrimu_gyros[:,0]):.6f}, max={np.max(np.abs(corrimu_gyros[:,0])):.6f}")
    print(f"  Y: mean={np.mean(corrimu_gyros[:,1]):+.6f}, std={np.std(corrimu_gyros[:,1]):.6f}, max={np.max(np.abs(corrimu_gyros[:,1])):.6f}")
    print(f"  Z: mean={np.mean(corrimu_gyros[:,2]):+.6f}, std={np.std(corrimu_gyros[:,2]):.6f}, max={np.max(np.abs(corrimu_gyros[:,2])):.6f}")
    print(f"  Magnitude: mean={np.mean(corrimu_gyro_mag):.6f}, std={np.std(corrimu_gyro_mag):.6f}, max={np.max(corrimu_gyro_mag):.6f}")

    print(f"\n/imu/data:")
    print(f"  X: mean={np.mean(imu_gyros[:,0]):+.6f}, std={np.std(imu_gyros[:,0]):.6f}, max={np.max(np.abs(imu_gyros[:,0])):.6f}")
    print(f"  Y: mean={np.mean(imu_gyros[:,1]):+.6f}, std={np.std(imu_gyros[:,1]):.6f}, max={np.max(np.abs(imu_gyros[:,1])):.6f}")
    print(f"  Z: mean={np.mean(imu_gyros[:,2]):+.6f}, std={np.std(imu_gyros[:,2]):.6f}, max={np.max(np.abs(imu_gyros[:,2])):.6f}")
    print(f"  Magnitude: mean={np.mean(imu_gyro_mag):.6f}, std={np.std(imu_gyro_mag):.6f}, max={np.max(imu_gyro_mag):.6f}")

    print("\n--- Extreme Values Count ---")
    for thresh in [15, 20, 30, 50]:
        corrimu_count = np.sum(corrimu_acc_mag > thresh)
        imu_count = np.sum(imu_acc_mag > thresh)
        print(f"Acc magnitude > {thresh} m/s²: CORRIMU={corrimu_count} ({100*corrimu_count/len(corrimu_acc_mag):.2f}%), /imu/data={imu_count} ({100*imu_count/len(imu_acc_mag):.2f}%)")

    print("\n" + "="*80)
    print("Output files:")
    print(f"  {OUTPUT_DIR}/imu_comparison_full.png")
    print(f"  {OUTPUT_DIR}/imu_comparison_zoomed.png")
    print(f"  {OUTPUT_DIR}/imu_comparison_diff.png")
    print("="*80)

if __name__ == "__main__":
    main()
