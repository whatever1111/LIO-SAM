#!/usr/bin/env python3
"""
分析GPS和Fusion轨迹在380秒后偏离的原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '/root/autodl-tmp/catkin_ws/src/LIO-SAM/output'

# 读取数据
print("Loading data...")
fusion_df = pd.read_csv(os.path.join(output_dir, 'fusion_trajectory.csv'))
gps_df = pd.read_csv(os.path.join(output_dir, 'gps_trajectory.csv'))
imu_gyro_df = pd.read_csv(os.path.join(output_dir, 'imu_angular_velocity.csv'))
imu_accel_df = pd.read_csv(os.path.join(output_dir, 'imu_linear_acceleration.csv'))
fusion_latency_df = pd.read_csv(os.path.join(output_dir, 'fusion_latency.csv'))

# 转换时间为相对时间（秒）
start_time = min(fusion_df['time'].min(), gps_df['time'].min())
fusion_df['rel_time'] = fusion_df['time'] - start_time
gps_df['rel_time'] = gps_df['time'] - start_time

print(f"Fusion time range: {fusion_df['rel_time'].min():.1f} - {fusion_df['rel_time'].max():.1f} s")
print(f"GPS time range: {gps_df['rel_time'].min():.1f} - {gps_df['rel_time'].max():.1f} s")

# 找到380秒附近的数据
diverge_time = 380
window_before = 50  # 分析380秒前50秒
window_after = 100  # 分析380秒后100秒

fusion_before = fusion_df[(fusion_df['rel_time'] >= diverge_time - window_before) &
                          (fusion_df['rel_time'] < diverge_time)]
fusion_after = fusion_df[(fusion_df['rel_time'] >= diverge_time) &
                         (fusion_df['rel_time'] < diverge_time + window_after)]

gps_before = gps_df[(gps_df['rel_time'] >= diverge_time - window_before) &
                    (gps_df['rel_time'] < diverge_time)]
gps_after = gps_df[(gps_df['rel_time'] >= diverge_time) &
                   (gps_df['rel_time'] < diverge_time + window_after)]

print(f"\nData points around {diverge_time}s:")
print(f"  Fusion before: {len(fusion_before)}, after: {len(fusion_after)}")
print(f"  GPS before: {len(gps_before)}, after: {len(gps_after)}")

# 计算每个时间点的偏差
# 对GPS数据进行插值以匹配fusion时间戳
from scipy.interpolate import interp1d

gps_x_interp = interp1d(gps_df['rel_time'], gps_df['x'], fill_value='extrapolate')
gps_y_interp = interp1d(gps_df['rel_time'], gps_df['y'], fill_value='extrapolate')
gps_z_interp = interp1d(gps_df['rel_time'], gps_df['z'], fill_value='extrapolate')

fusion_df['gps_x'] = gps_x_interp(fusion_df['rel_time'])
fusion_df['gps_y'] = gps_y_interp(fusion_df['rel_time'])
fusion_df['gps_z'] = gps_z_interp(fusion_df['rel_time'])

fusion_df['error_x'] = fusion_df['x'] - fusion_df['gps_x']
fusion_df['error_y'] = fusion_df['y'] - fusion_df['gps_y']
fusion_df['error_z'] = fusion_df['z'] - fusion_df['gps_z']
fusion_df['error_total'] = np.sqrt(fusion_df['error_x']**2 + fusion_df['error_y']**2 + fusion_df['error_z']**2)

# 分析偏差变化
print("\n=== Error Analysis ===")
for t in [100, 200, 300, 380, 400, 500, 600, 700, 800]:
    data_at_t = fusion_df[np.abs(fusion_df['rel_time'] - t) < 1]
    if len(data_at_t) > 0:
        err = data_at_t['error_total'].mean()
        err_x = data_at_t['error_x'].mean()
        err_y = data_at_t['error_y'].mean()
        err_z = data_at_t['error_z'].mean()
        print(f"  t={t:3d}s: total={err:8.2f}m, x={err_x:8.2f}m, y={err_y:8.2f}m, z={err_z:8.2f}m")

# 计算偏差增长率
print("\n=== Error Growth Rate ===")
time_points = [350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
errors = []
for t in time_points:
    data_at_t = fusion_df[np.abs(fusion_df['rel_time'] - t) < 5]
    if len(data_at_t) > 0:
        errors.append(data_at_t['error_total'].mean())
    else:
        errors.append(np.nan)

for i in range(1, len(time_points)):
    if not np.isnan(errors[i]) and not np.isnan(errors[i-1]):
        dt = time_points[i] - time_points[i-1]
        de = errors[i] - errors[i-1]
        rate = de / dt
        print(f"  {time_points[i-1]}-{time_points[i]}s: error growth = {rate:.3f} m/s")

# 分析Y轴差异（图中显示Y轴也有明显差异）
print("\n=== Y-axis Analysis (GPS vs Fusion) ===")
for t in [100, 200, 300, 400, 500, 600, 700, 800]:
    fusion_at_t = fusion_df[np.abs(fusion_df['rel_time'] - t) < 1]
    gps_at_t = gps_df[np.abs(gps_df['rel_time'] - t) < 1]
    if len(fusion_at_t) > 0 and len(gps_at_t) > 0:
        f_y = fusion_at_t['y'].mean()
        g_y = gps_at_t['y'].mean()
        print(f"  t={t:3d}s: Fusion Y={f_y:8.2f}m, GPS Y={g_y:8.2f}m, diff={f_y-g_y:8.2f}m")

# 计算速度
fusion_df['vx'] = fusion_df['x'].diff() / fusion_df['rel_time'].diff()
fusion_df['vy'] = fusion_df['y'].diff() / fusion_df['rel_time'].diff()
fusion_df['vz'] = fusion_df['z'].diff() / fusion_df['rel_time'].diff()
fusion_df['speed'] = np.sqrt(fusion_df['vx']**2 + fusion_df['vy']**2 + fusion_df['vz']**2)

gps_df['vx'] = gps_df['x'].diff() / gps_df['rel_time'].diff()
gps_df['vy'] = gps_df['y'].diff() / gps_df['rel_time'].diff()
gps_df['vz'] = gps_df['z'].diff() / gps_df['rel_time'].diff()
gps_df['speed'] = np.sqrt(gps_df['vx']**2 + gps_df['vy']**2 + gps_df['vz']**2)

print("\n=== Speed Comparison ===")
for t in [300, 350, 380, 400, 450, 500]:
    fusion_at_t = fusion_df[np.abs(fusion_df['rel_time'] - t) < 5]
    gps_at_t = gps_df[np.abs(gps_df['rel_time'] - t) < 5]
    if len(fusion_at_t) > 0 and len(gps_at_t) > 0:
        f_speed = fusion_at_t['speed'].median()
        g_speed = gps_at_t['speed'].median()
        print(f"  t={t:3d}s: Fusion speed={f_speed:.2f}m/s, GPS speed={g_speed:.2f}m/s")

# 分析IMU数据
print("\n=== IMU Analysis around divergence point ===")
# IMU数据时间范围
imu_start = imu_gyro_df['relative_time_s'].min()
imu_end = imu_gyro_df['relative_time_s'].max()
print(f"IMU data range: {imu_start:.1f} - {imu_end:.1f}s")

# 计算角速度magnitude
imu_gyro_df['angular_mag'] = np.sqrt(imu_gyro_df['wx_rad_s']**2 + imu_gyro_df['wy_rad_s']**2 + imu_gyro_df['wz_rad_s']**2)

# 检查380秒附近是否有异常旋转
for t_start, t_end in [(330, 380), (380, 430), (430, 480)]:
    imu_window = imu_gyro_df[(imu_gyro_df['relative_time_s'] >= t_start) &
                            (imu_gyro_df['relative_time_s'] < t_end)]
    if len(imu_window) > 0:
        max_ang = imu_window['angular_mag'].max()
        mean_ang = imu_window['angular_mag'].mean()
        print(f"  {t_start}-{t_end}s: max angular vel = {max_ang:.4f} rad/s, mean = {mean_ang:.4f} rad/s")

# 检查加速度异常
imu_accel_df['accel_mag'] = np.sqrt(imu_accel_df['ax_m_s2']**2 + imu_accel_df['ay_m_s2']**2 + imu_accel_df['az_m_s2']**2)

print("\n=== Acceleration Analysis ===")
for t_start, t_end in [(330, 380), (380, 430), (430, 480)]:
    imu_window = imu_accel_df[(imu_accel_df['relative_time_s'] >= t_start) &
                              (imu_accel_df['relative_time_s'] < t_end)]
    if len(imu_window) > 0:
        mean_accel = imu_window['accel_mag'].mean()
        std_accel = imu_window['accel_mag'].std()
        print(f"  {t_start}-{t_end}s: mean accel = {mean_accel:.4f} m/s^2, std = {std_accel:.4f}")

# 分析延迟
print("\n=== Latency Analysis ===")
latency_before = fusion_latency_df[(fusion_latency_df['relative_time_s'] >= diverge_time - window_before) &
                                    (fusion_latency_df['relative_time_s'] < diverge_time)]
latency_after = fusion_latency_df[(fusion_latency_df['relative_time_s'] >= diverge_time) &
                                   (fusion_latency_df['relative_time_s'] < diverge_time + window_after)]

if len(latency_before) > 0:
    print(f"  Before {diverge_time}s: mean={latency_before['latency_ms'].mean():.1f}ms, max={latency_before['latency_ms'].max():.1f}ms")
if len(latency_after) > 0:
    print(f"  After {diverge_time}s: mean={latency_after['latency_ms'].mean():.1f}ms, max={latency_after['latency_ms'].max():.1f}ms")

# 创建详细分析图
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

# 1. X位置对比 (放大380秒附近)
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(fusion_df['rel_time'], fusion_df['x'], 'b-', label='Fusion', linewidth=0.5)
ax1.plot(gps_df['rel_time'], gps_df['x'], 'r-', label='GPS', linewidth=0.5)
ax1.axvline(x=380, color='green', linestyle='--', label='Divergence start')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('X Position (m)')
ax1.set_title('X Position: GPS vs Fusion')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([300, 600])

# 2. Y位置对比
ax2 = fig.add_subplot(gs[0, 2:4])
ax2.plot(fusion_df['rel_time'], fusion_df['y'], 'b-', label='Fusion', linewidth=0.5)
ax2.plot(gps_df['rel_time'], gps_df['y'], 'r-', label='GPS', linewidth=0.5)
ax2.axvline(x=380, color='green', linestyle='--', label='Divergence start')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Y Position (m)')
ax2.set_title('Y Position: GPS vs Fusion')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([300, 600])

# 3. 位置误差随时间变化
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(fusion_df['rel_time'], fusion_df['error_x'], 'r-', label='X error', linewidth=0.5, alpha=0.7)
ax3.plot(fusion_df['rel_time'], fusion_df['error_y'], 'g-', label='Y error', linewidth=0.5, alpha=0.7)
ax3.plot(fusion_df['rel_time'], fusion_df['error_total'], 'b-', label='Total error', linewidth=0.8)
ax3.axvline(x=380, color='green', linestyle='--', alpha=0.5)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Error (m)')
ax3.set_title('Position Error Over Time (Fusion - GPS)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 误差增长分析
ax4 = fig.add_subplot(gs[1, 2:4])
# 计算滑动窗口内的误差
window_size = 50  # 10秒窗口
fusion_df['error_smooth'] = fusion_df['error_total'].rolling(window=window_size, center=True).mean()
ax4.plot(fusion_df['rel_time'], fusion_df['error_smooth'], 'b-', linewidth=1)
ax4.axvline(x=380, color='green', linestyle='--', label='Divergence start')

# 标记关键时间点
key_times = [200, 300, 380, 500, 600, 700, 800]
for t in key_times:
    data = fusion_df[np.abs(fusion_df['rel_time'] - t) < 5]
    if len(data) > 0:
        err = data['error_smooth'].mean()
        if not np.isnan(err):
            ax4.scatter([t], [err], s=50, zorder=5)
            ax4.annotate(f'{t}s: {err:.1f}m', (t, err), textcoords="offset points", xytext=(0,10))

ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Smoothed Error (m)')
ax4.set_title('Error Growth Analysis')
ax4.grid(True, alpha=0.3)

# 5. 速度对比
ax5 = fig.add_subplot(gs[2, 0:2])
fusion_speed_smooth = fusion_df['speed'].rolling(window=20).median()
gps_speed_smooth = gps_df['speed'].rolling(window=20).median()
ax5.plot(fusion_df['rel_time'], fusion_speed_smooth, 'b-', label='Fusion', linewidth=0.5)
ax5.plot(gps_df['rel_time'], gps_speed_smooth, 'r-', label='GPS', linewidth=0.5)
ax5.axvline(x=380, color='green', linestyle='--', alpha=0.5)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Speed (m/s)')
ax5.set_title('Speed Comparison')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlim([300, 600])
ax5.set_ylim([0, 20])

# 6. IMU角速度
ax6 = fig.add_subplot(gs[2, 2:4])
imu_plot = imu_gyro_df[(imu_gyro_df['relative_time_s'] >= 300) & (imu_gyro_df['relative_time_s'] <= 600)]
if len(imu_plot) > 0:
    ax6.plot(imu_plot['relative_time_s'], imu_plot['wz_rad_s'], 'b-', linewidth=0.3, alpha=0.7, label='wz (yaw rate)')
else:
    ax6.text(0.5, 0.5, 'No IMU data in this range', ha='center', va='center', transform=ax6.transAxes)
ax6.axvline(x=380, color='green', linestyle='--', alpha=0.5)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Angular Velocity (rad/s)')
ax6.set_title('Yaw Rate (wz) Around Divergence Point')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. 2D轨迹对比（300-500秒）
ax7 = fig.add_subplot(gs[3, 0:2])
fusion_plot = fusion_df[(fusion_df['rel_time'] >= 300) & (fusion_df['rel_time'] <= 500)]
gps_plot = gps_df[(gps_df['rel_time'] >= 300) & (gps_df['rel_time'] <= 500)]

# 标记380��位置
fusion_380 = fusion_df[np.abs(fusion_df['rel_time'] - 380) < 1]
gps_380 = gps_df[np.abs(gps_df['rel_time'] - 380) < 1]

ax7.plot(fusion_plot['x'], fusion_plot['y'], 'b-', label='Fusion', linewidth=1)
ax7.plot(gps_plot['x'], gps_plot['y'], 'r-', label='GPS', linewidth=1)
if len(fusion_380) > 0:
    ax7.scatter(fusion_380['x'].mean(), fusion_380['y'].mean(), c='blue', s=100, marker='*', zorder=5)
if len(gps_380) > 0:
    ax7.scatter(gps_380['x'].mean(), gps_380['y'].mean(), c='red', s=100, marker='*', zorder=5)
ax7.set_xlabel('X (m)')
ax7.set_ylabel('Y (m)')
ax7.set_title('2D Trajectory (300-500s) - Stars mark t=380s')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.axis('equal')

# 8. 处理延迟
ax8 = fig.add_subplot(gs[3, 2:4])
latency_plot = fusion_latency_df[(fusion_latency_df['relative_time_s'] >= 300) &
                                  (fusion_latency_df['relative_time_s'] <= 600)]
ax8.plot(latency_plot['relative_time_s'], latency_plot['latency_ms'], 'b-', linewidth=0.5)
ax8.axvline(x=380, color='green', linestyle='--', alpha=0.5, label='Divergence start')
ax8.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='400ms threshold')
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Latency (ms)')
ax8.set_title('LIO-SAM Processing Latency')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.suptitle('GPS vs Fusion Divergence Analysis at t=380s', fontsize=14, y=1.01)
plt.savefig(os.path.join(output_dir, 'divergence_analysis_380s.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {os.path.join(output_dir, 'divergence_analysis_380s.png')}")

# 进一步分析：计算航向角差异
print("\n=== Heading Analysis ===")
# 计算航向角（从速度向量）
fusion_df['heading'] = np.arctan2(fusion_df['vy'], fusion_df['vx']) * 180 / np.pi
gps_df['heading'] = np.arctan2(gps_df['vy'], gps_df['vx']) * 180 / np.pi

for t in [300, 350, 380, 400, 450, 500]:
    fusion_at_t = fusion_df[(fusion_df['rel_time'] >= t-2) & (fusion_df['rel_time'] <= t+2)]
    gps_at_t = gps_df[(gps_df['rel_time'] >= t-2) & (gps_df['rel_time'] <= t+2)]
    if len(fusion_at_t) > 0 and len(gps_at_t) > 0:
        f_heading = fusion_at_t['heading'].median()
        g_heading = gps_at_t['heading'].median()
        diff = f_heading - g_heading
        # 规范化到-180到180
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        print(f"  t={t:3d}s: Fusion heading={f_heading:7.2f}°, GPS heading={g_heading:7.2f}°, diff={diff:7.2f}°")

# 关键结论
print("\n" + "="*60)
print("DIVERGENCE ANALYSIS SUMMARY")
print("="*60)

# 计算380秒前后的误差增长率
err_at_350 = fusion_df[np.abs(fusion_df['rel_time'] - 350) < 5]['error_total'].mean()
err_at_380 = fusion_df[np.abs(fusion_df['rel_time'] - 380) < 5]['error_total'].mean()
err_at_500 = fusion_df[np.abs(fusion_df['rel_time'] - 500) < 5]['error_total'].mean()
err_at_700 = fusion_df[np.abs(fusion_df['rel_time'] - 700) < 5]['error_total'].mean()

print(f"\n1. Error at key time points:")
print(f"   - t=350s: {err_at_350:.2f}m")
print(f"   - t=380s: {err_at_380:.2f}m (divergence start)")
print(f"   - t=500s: {err_at_500:.2f}m")
print(f"   - t=700s: {err_at_700:.2f}m")

if err_at_500 - err_at_380 > 0:
    growth_rate = (err_at_500 - err_at_380) / (500 - 380)
    print(f"\n2. Error growth rate after divergence: {growth_rate:.4f} m/s ({growth_rate*60:.2f} m/min)")

# 检查主要是X还是Y方向偏差
err_x_500 = fusion_df[np.abs(fusion_df['rel_time'] - 500) < 5]['error_x'].mean()
err_y_500 = fusion_df[np.abs(fusion_df['rel_time'] - 500) < 5]['error_y'].mean()
print(f"\n3. Error direction at t=500s:")
print(f"   - X error: {err_x_500:.2f}m")
print(f"   - Y error: {err_y_500:.2f}m")
if abs(err_x_500) > abs(err_y_500):
    print(f"   => X error is dominant ({abs(err_x_500)/abs(err_y_500):.1f}x larger)")
else:
    print(f"   => Y error is dominant ({abs(err_y_500)/abs(err_x_500):.1f}x larger)")

print("\n4. Possible causes:")
print("   - IMU drift accumulation (expected in LIO without GPS correction)")
print("   - LiDAR-based odometry drift in feature-sparse areas")
print("   - GPS data not being properly fused into optimization")
print("   - Heading/yaw error accumulation causing X-Y position divergence")

plt.show()
