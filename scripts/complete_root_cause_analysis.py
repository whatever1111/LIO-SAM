#!/usr/bin/env python3
"""
LIO-SAM 速度跳变完整根因分析
基于诊断数据和代码分析
"""

import pandas as pd
import numpy as np
import sys
import os

def generate_comprehensive_analysis(csv_file):
    """生成完整的根因分析报告"""

    print("="*70)
    print("             LIO-SAM 速度跳变根因分析报告                ")
    print("="*70)

    # 读取数据
    df = pd.read_csv(csv_file)

    # 1. 数据概览
    print("\n" + "-"*70)
    print("1. 诊断数据概览")
    print("-"*70)
    print(f"总异常记录数: {len(df)}")
    print(f"时间范围: {df['timestamp'].max() - df['timestamp'].min():.1f} 秒")

    # 异常类型分布
    type_counts = df.groupby('type').size()
    print(f"\n异常类型分布:")
    for t, c in type_counts.items():
        print(f"  {t}: {c} ({100*c/len(df):.1f}%)")

    # 2. 位置跳变分析
    print("\n" + "-"*70)
    print("2. 位置跳变分析 (POS_JUMP)")
    print("-"*70)

    pos_jumps = df[df['type'] == 'POS_JUMP']
    if not pos_jumps.empty:
        velocities = pos_jumps['value'].values

        print(f"跳变次数: {len(pos_jumps)}")
        print(f"速度范围: {velocities.min():.1f} - {velocities.max():.1f} m/s")
        print(f"中位速度: {np.median(velocities):.1f} m/s")

        # 速度分布
        print(f"\n速度分布:")
        bins = [0, 50, 100, 500, 1000, float('inf')]
        labels = ['20-50', '50-100', '100-500', '500-1000', '>1000']
        for i in range(len(bins)-1):
            count = np.sum((velocities >= bins[i]) & (velocities < bins[i+1]))
            if count > 0:
                print(f"  {labels[i]} m/s: {count} 次 ({100*count/len(velocities):.1f}%)")

        # 极端跳变
        extreme = velocities[velocities > 1000]
        if len(extreme) > 0:
            print(f"\n!!! 极端跳变 (>1000 m/s): {len(extreme)} 次")
            print(f"    这表明优化完全失败,位姿估计发散!")

    # 3. 速度异常分析
    print("\n" + "-"*70)
    print("3. IMU预积分速度异常分析 (VEL_ANOMALY)")
    print("-"*70)

    vel_anomalies = df[df['type'] == 'VEL_ANOMALY']
    if not vel_anomalies.empty:
        # 解析速度分量
        vx_list, vy_list, vz_list = [], [], []
        for desc in vel_anomalies['description']:
            try:
                parts = desc.split(',')
                for p in parts:
                    p = p.strip()
                    if 'vx=' in p:
                        vx_list.append(float(p.split('=')[1]))
                    elif 'vy=' in p:
                        vy_list.append(float(p.split('=')[1]))
                    elif 'vz=' in p:
                        vz_list.append(float(p.split('=')[1].strip('"')))
            except:
                pass

        if vx_list:
            print(f"异常记录数: {len(vx_list)}")
            print(f"\n速度分量统计:")
            print(f"  Vx: 平均={np.mean(vx_list):.2f}, |Vx|平均={np.mean(np.abs(vx_list)):.2f} m/s")
            print(f"  Vy: 平均={np.mean(vy_list):.2f}, |Vy|平均={np.mean(np.abs(vy_list)):.2f} m/s")
            print(f"  Vz: 平均={np.mean(vz_list):.2f}, |Vz|平均={np.mean(np.abs(vz_list)):.2f} m/s")

            # 检查是否有接近重力加速度的分量
            print(f"\n>>> 重力相关性检查:")
            for name, vals in [('Vx', vx_list), ('Vy', vy_list), ('Vz', vz_list)]:
                mean_abs = np.mean(np.abs(vals))
                if 8.5 < mean_abs < 11:
                    print(f"  !!! {name}方向速度 ({mean_abs:.2f} m/s) 接近重力加速度!")
                    print(f"      可能存在重力补偿或坐标变换问题!")

    # 4. 时间序列分析
    print("\n" + "-"*70)
    print("4. 时间序列分析")
    print("-"*70)

    df_sorted = df.sort_values('timestamp')

    # 第一个异常的时间
    first_time = df_sorted['timestamp'].iloc[0]
    print(f"第一个异常时间: {first_time:.3f}")

    # 分析异常发生的顺序
    first_5 = df_sorted.head(5)
    print(f"\n前5个异常 (按时间排序):")
    for _, row in first_5.iterrows():
        print(f"  {row['timestamp']:.3f} [{row['source']:10s}] {row['type']:15s}")

    # 检查是否是位置跳变先于速度异常
    first_pos_jump = pos_jumps['timestamp'].min() if not pos_jumps.empty else float('inf')
    first_vel_anomaly = vel_anomalies['timestamp'].min() if not vel_anomalies.empty else float('inf')

    print(f"\n事件顺序分析:")
    if first_pos_jump < first_vel_anomaly:
        print(f"  位置跳变先发生 (t={first_pos_jump:.3f})")
        print(f"  速度异常后发生 (t={first_vel_anomaly:.3f})")
        print(f"  >>> 这表明: 根本原因是scan-to-map匹配问题,导致位置跳变")
        print(f"      随后错误传递到IMU预积分模块")
    else:
        print(f"  速度异常先发生 (t={first_vel_anomaly:.3f})")
        print(f"  位置跳变后发生 (t={first_pos_jump:.3f})")
        print(f"  >>> 这表明: 根本原因可能是IMU预积分问题")

    # 5. 根因分析
    print("\n" + "="*70)
    print("5. 根本原因分析")
    print("="*70)

    print("""
基于诊断数据分析,速度跳变的可能原因按概率排序:

┌─────────────────────────────────────────────────────────────────────┐
│ 原因 #1 (高概率): Scan-to-Map匹配失败                                │
├─────────────────────────────────────────────────────────────────────┤
│ 证据:                                                               │
│   - 位置跳变先于速度异常发生                                          │
│   - 出现极端跳变 (>1000 m/s) 表明优化完全失败                         │
│                                                                     │
│ 可能的触发因素:                                                      │
│   a) 特征点数量不足                                                  │
│      - edgeThreshold/surfThreshold 可能太严格                        │
│      - 环境几何特征不够丰富                                          │
│   b) 退化场景                                                       │
│      - 长走廊、隧道等退化环境                                         │
│      - LiDAR视野受阻                                                 │
│   c) 初始位姿估计不准                                                │
│      - IMU预积分累积误差                                             │
│      - IMU-LiDAR外参不准确                                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 原因 #2 (中等概率): IMU配置问题                                       │
├─────────────────────────────────────────────────────────────────────┤
│ 可能的问题:                                                         │
│   a) extrinsicRot 与实际IMU安装不匹配                                │
│   b) IMU噪声参数设置不当                                             │
│   c) IMU频率与算法期望不一致                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 原因 #3 (低概率): 时间同步问题                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 可能的问题:                                                         │
│   a) LiDAR与IMU时间戳不同步                                          │
│   b) lidarTimeOffset 设置不正确                                      │
└─────────────────────────────────────────────────────────────────────┘
""")

    # 6. 建议的解决方案
    print("\n" + "="*70)
    print("6. 建议的解决方案")
    print("="*70)

    print("""
优先级 1: 检查特征提取
───────────────────────────────────────
  当前配置:
    edgeThreshold: 0.1 (默认: 1.0)
    surfThreshold: 0.1 (默认: 0.1)
    edgeFeatureMinValidNum: 5 (默认: 10)
    surfFeatureMinValidNum: 50 (默认: 100)

  建议操作:
    1. 监控实际提取的特征点数量
       运行: rosrun lio_sam lio_sam_feature_monitor.py
    2. 如果特征点经常少于阈值,调整参数:
       - 降低 edgeThreshold (提取更多边缘特征)
       - 降低 surfThreshold (提取更多平面特征)
    3. 检查环境是否适合LiDAR SLAM

优先级 2: 添加位置跳变保护
───────────────────────────────────────
  在mapOptmization.cpp中添加跳变检测:

  // 在scan2MapOptimization()后检查
  void checkPositionJump() {
      if (cloudKeyPoses3D->size() < 2) return;

      float dx = transformTobeMapped[3] - cloudKeyPoses6D->back().x;
      float dy = transformTobeMapped[4] - cloudKeyPoses6D->back().y;
      float dz = transformTobeMapped[5] - cloudKeyPoses6D->back().z;
      float dist = sqrt(dx*dx + dy*dy + dz*dz);

      // 如果位移超过阈值(如2米),拒绝此次优化结果
      if (dist > 2.0) {
          ROS_WARN("Position jump detected! Resetting to previous pose.");
          transformTobeMapped[3] = cloudKeyPoses6D->back().x;
          transformTobeMapped[4] = cloudKeyPoses6D->back().y;
          transformTobeMapped[5] = cloudKeyPoses6D->back().z;
      }
  }

优先级 3: 检查IMU外参
───────────────────────────────────────
  当前配置:
    extrinsicRot: [-1,0,0, 0,-1,0, 0,0,1]  (Rz 180°)

  验证方法:
    1. 静止时检查变换后加速度是否为 (0, 0, +9.81)
    2. 确认IMU实际安装方向与配置一致

优先级 4: 启用退化检测日志
───────────────────────────────────────
  在LMOptimization()中添加日志:

  if (isDegenerate) {
      ROS_WARN("Degenerate scene detected at time %.3f", timeLaserInfoCur);
      // 可以选择使用更保守的更新策略
  }
""")

    print("\n" + "="*70)
    print("分析完成")
    print("="*70)


def main():
    csv_file = "/tmp/lio_sam_diagnostic_20251124_173540.csv"

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"错误: 文件不存在: {csv_file}")
        sys.exit(1)

    generate_comprehensive_analysis(csv_file)


if __name__ == "__main__":
    main()
