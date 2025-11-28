#!/usr/bin/env python3
"""
分析诊断监控器生成的CSV数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

def analyze_diagnostic_data(csv_file):
    """分析诊断数据"""
    print(f"\n{'='*60}")
    print("LIO-SAM Diagnostic Data Analysis")
    print(f"{'='*60}")
    print(f"Analyzing: {csv_file}\n")

    # 读取CSV数据
    df = pd.read_csv(csv_file)

    if df.empty:
        print("No data found in CSV file!")
        return

    # 基本统计
    print("Data Overview:")
    print(f"  Total records: {len(df)}")
    print(f"  Time range: {df['timestamp'].min():.2f} to {df['timestamp'].max():.2f}")
    print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()):.2f} seconds")
    print()

    # 按数据源分组统计
    print("Anomalies by Source:")
    source_counts = df.groupby('source')['type'].count()
    for source, count in source_counts.items():
        print(f"  {source:15s}: {count:4d} events")
    print()

    # 按异常类型分组统计
    print("Anomalies by Type:")
    type_counts = df.groupby('type')['source'].count()
    for anomaly_type, count in type_counts.items():
        print(f"  {anomaly_type:20s}: {count:4d} events")
    print()

    # 分析速度异常
    vel_anomalies = df[df['type'].str.contains('VEL_ANOMALY', na=False)]
    if not vel_anomalies.empty:
        print("Velocity Anomalies Analysis:")
        print(f"  Total velocity anomalies: {len(vel_anomalies)}")
        print(f"  Max velocity: {vel_anomalies['value'].max():.2f} m/s")
        print(f"  Mean velocity during anomaly: {vel_anomalies['value'].mean():.2f} m/s")
        print()

        # 找出最严重的速度异常
        worst_vel = vel_anomalies.loc[vel_anomalies['value'].idxmax()]
        print(f"  Worst velocity anomaly:")
        print(f"    Time: {worst_vel['timestamp']:.3f}")
        print(f"    Source: {worst_vel['source']}")
        print(f"    Value: {worst_vel['value']:.2f} m/s")
        print(f"    Description: {worst_vel['description']}")
        print()

    # 分析IMU异常
    imu_anomalies = df[df['source'] == 'IMU']
    if not imu_anomalies.empty:
        print("IMU Anomalies Analysis:")
        acc_anomalies = imu_anomalies[imu_anomalies['type'] == 'ACC_ANOMALY']
        gyro_anomalies = imu_anomalies[imu_anomalies['type'] == 'GYRO_ANOMALY']
        print(f"  Acceleration anomalies: {len(acc_anomalies)}")
        if not acc_anomalies.empty:
            print(f"    Max acceleration: {acc_anomalies['value'].max():.2f} m/s^2")
        print(f"  Gyroscope anomalies: {len(gyro_anomalies)}")
        if not gyro_anomalies.empty:
            print(f"    Max angular rate: {gyro_anomalies['value'].max():.2f} rad/s")
        print()

    # 分析LiDAR异常
    lidar_anomalies = df[df['source'] == 'LIDAR']
    if not lidar_anomalies.empty:
        print("LiDAR Anomalies Analysis:")
        sparse_clouds = lidar_anomalies[lidar_anomalies['type'] == 'SPARSE_CLOUD']
        range_anomalies = lidar_anomalies[lidar_anomalies['type'] == 'RANGE_ANOMALY']
        print(f"  Sparse cloud events: {len(sparse_clouds)}")
        print(f"  Range anomalies: {len(range_anomalies)}")
        if not range_anomalies.empty:
            print(f"    Max range detected: {range_anomalies['value'].max():.2f} m")
        print()

    # 时间相关性分析
    print("Temporal Correlation Analysis:")

    # 查找时间上接近的异常
    df_sorted = df.sort_values('timestamp')
    df_sorted['time_diff'] = df_sorted['timestamp'].diff()

    # 找出1秒内发生的多个异常
    clusters = []
    current_cluster = []

    for idx, row in df_sorted.iterrows():
        if current_cluster and row['time_diff'] > 1.0:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [row]
        else:
            current_cluster.append(row)

    if len(current_cluster) > 1:
        clusters.append(current_cluster)

    print(f"  Found {len(clusters)} anomaly clusters (multiple anomalies within 1 second)")

    if clusters:
        print("\n  Significant Clusters:")
        for i, cluster in enumerate(clusters[:5]):  # 显示前5个聚类
            sources = set([row['source'] for row in cluster])
            types = set([row['type'] for row in cluster])
            print(f"\n  Cluster {i+1}:")
            print(f"    Time: {cluster[0]['timestamp']:.3f}")
            print(f"    Events: {len(cluster)}")
            print(f"    Sources: {', '.join(sources)}")
            print(f"    Types: {', '.join(types)}")

    # 根因分析
    print("\n" + "="*60)
    print("Root Cause Analysis:")
    print("="*60)

    # 分析IMU预积分异常
    imu_preint_anomalies = df[df['source'] == 'IMU_PREINT']
    if not imu_preint_anomalies.empty:
        print("\nIMU Preintegration velocity anomalies detected!")
        print(f"Total: {len(imu_preint_anomalies)} events")

        # 检查这些异常前后的其他传感器状态
        for idx, anomaly in imu_preint_anomalies.iterrows():
            time_window = 0.5  # 前后0.5秒
            related_events = df[
                (df['timestamp'] >= anomaly['timestamp'] - time_window) &
                (df['timestamp'] <= anomaly['timestamp'] + time_window) &
                (df.index != idx)
            ]

            if not related_events.empty:
                print(f"\n  Anomaly at {anomaly['timestamp']:.3f}s (vel={anomaly['value']:.2f} m/s):")
                related_by_source = related_events.groupby('source')['type'].count()
                for source, count in related_by_source.items():
                    print(f"    {source}: {count} related events")

    # 诊断结论
    print("\n" + "="*60)
    print("Diagnostic Conclusion:")
    print("="*60)

    # 基于数据分析得出结论
    imu_score = len(df[df['source'] == 'IMU']) / max(len(df), 1) * 100
    lidar_score = len(df[df['source'] == 'LIDAR']) / max(len(df), 1) * 100
    lio_score = (len(df[df['source'] == 'LIO_ODOM']) + len(df[df['source'] == 'IMU_PREINT'])) / max(len(df), 1) * 100

    print(f"\nProblem Distribution:")
    print(f"  IMU-related: {imu_score:.1f}%")
    print(f"  LiDAR-related: {lidar_score:.1f}%")
    print(f"  LIO algorithm-related: {lio_score:.1f}%")

    # 主要原因判断
    print("\nPrimary Cause Assessment:")
    if imu_score > 50:
        print("  >> IMU sensor is the likely primary cause")
        print("     Recommendations:")
        print("     - Check IMU calibration parameters")
        print("     - Verify IMU mounting and vibration isolation")
        print("     - Check for electromagnetic interference")
    elif lidar_score > 50:
        print("  >> LiDAR sensor is the likely primary cause")
        print("     Recommendations:")
        print("     - Check LiDAR point cloud quality")
        print("     - Verify LiDAR calibration")
        print("     - Check for environmental factors (reflective surfaces, etc.)")
    elif lio_score > 50:
        print("  >> LIO-SAM algorithm issues are likely")
        print("     Recommendations:")
        print("     - Review IMU-LiDAR extrinsic calibration")
        print("     - Check time synchronization between sensors")
        print("     - Adjust algorithm parameters (noise models, thresholds)")
    else:
        print("  >> Mixed or intermittent issues detected")
        print("     Recommendations:")
        print("     - Check overall system timing and synchronization")
        print("     - Review sensor data quality during anomaly periods")
        print("     - Consider environmental factors")

    # 生成报告文件
    report_file = csv_file.replace('.csv', '_analysis.txt')
    with open(report_file, 'w') as f:
        # 重定向print输出到文件
        original_stdout = sys.stdout
        sys.stdout = f

        # 重新运行所有分析输出
        print(f"\nLIO-SAM Diagnostic Analysis Report")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data file: {csv_file}")
        print(f"\nTotal anomalies: {len(df)}")
        print(f"IMU anomalies: {len(df[df['source'] == 'IMU'])}")
        print(f"LiDAR anomalies: {len(df[df['source'] == 'LIDAR'])}")
        print(f"LIO anomalies: {len(df[(df['source'] == 'LIO_ODOM') | (df['source'] == 'IMU_PREINT')])}")

        sys.stdout = original_stdout

    print(f"\nAnalysis report saved to: {report_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_diagnostic.py <csv_file>")
        print("Example: python3 analyze_diagnostic.py /tmp/lio_sam_diagnostic_20240101_120000.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    analyze_diagnostic_data(csv_file)

if __name__ == "__main__":
    main()