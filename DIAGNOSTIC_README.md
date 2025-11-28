# LIO-SAM 诊断监控系统

## 概述

这个诊断系统用于精确定位 LIO-SAM 运行中出现的异常高速度问题（例如 188492.94 m/s）。系统会监控并分析 IMU、LiDAR 和 LIO-SAM 算法的输出，帮助确定问题的根源。

## 系统组成

1. **diagnostic_monitor.py** - 实时监控脚本
   - 监控 IMU 原始数据 (/imu/data)
   - 监控 LiDAR 点云 (/lidar_points)
   - 监控 LIO-SAM 里程计输出 (lio_sam/mapping/odometry)
   - 监控 IMU 预积分输出 (odometry/imu)
   - 记录异常事件到 CSV 和日志文件

2. **analyze_diagnostic.py** - 数据分析脚本
   - 分析 CSV 数据中的异常模式
   - 执行根因分析
   - 生成诊断报告

3. **visualize_diagnostic.py** - 实时可视化脚本
   - 实时显示传感器数据
   - 显示异常事件时间线
   - 提供关键指标的可视化

4. **run_diagnostic.sh** - 主启动脚本
   - 提供统一的启动接口
   - 自动化诊断流程

## 使用方法

### 1. 运行完整诊断（推荐）

```bash
# 给脚本添加执行权限
chmod +x /root/autodl-tmp/catkin_ws/src/LIO-SAM/run_diagnostic.sh

# 运行诊断监控 + 评估脚本
./run_diagnostic.sh monitor
```

这将：
- 启动诊断监控器
- 运行你的 run_evaluation.sh 脚本
- 实时监控所有相关 topics
- 在运行结束后自动分析数据

### 2. 单独运行监控器

```bash
# 启动监控器
python3 /root/autodl-tmp/catkin_ws/src/LIO-SAM/scripts/diagnostic_monitor.py

# 在另一个终端运行你的评估脚本
./run_evaluation.sh
```

### 3. 分析已有数据

```bash
# 分析特定的 CSV 文件
./run_diagnostic.sh analyze /tmp/lio_sam_diagnostic_20241124_xxxxx.csv

# 或直接运行
python3 scripts/analyze_diagnostic.py /tmp/lio_sam_diagnostic_20241124_xxxxx.csv
```

### 4. 实时可视化

```bash
# 可视化诊断数据
./run_diagnostic.sh visualize /tmp/lio_sam_diagnostic_20241124_xxxxx.csv

# 或直接运行
python3 scripts/visualize_diagnostic.py /tmp/lio_sam_diagnostic_20241124_xxxxx.csv
```

## 输出文件

监控器会生成以下文件：

1. **日志文件**: `/tmp/lio_sam_diagnostic_YYYYMMDD_HHMMSS.txt`
   - 包含所有检测到的异常的详细信息
   - 包含实时统计信息

2. **CSV 文件**: `/tmp/lio_sam_diagnostic_YYYYMMDD_HHMMSS.csv`
   - 结构化的异常事件数据
   - 用于后续分析和可视化

3. **分析报告**: `/tmp/lio_sam_diagnostic_YYYYMMDD_HHMMSS_analysis.txt`
   - 由分析脚本生成
   - 包含根因分析和建议

## 异常检测阈值

系统使用以下默认阈值（可在 diagnostic_monitor.py 中调整）：

- **速度阈值**: 10 m/s
- **加速度阈值**: 50 m/s²
- **角速度阈值**: 10 rad/s
- **最小点云数量**: 100 点
- **最大测距**: 200 m

## 诊断结果解读

### 问题来源判断

系统会根据异常分布判断问题来源：

1. **IMU 问题** (>50% IMU 异常)
   - 检查 IMU 标定参数
   - 验证 IMU 安装和减震
   - 检查电磁干扰

2. **LiDAR 问题** (>50% LiDAR 异常)
   - 检查点云质量
   - 验证 LiDAR 标定
   - 检查环境因素（反射表面等）

3. **LIO-SAM 算法问题** (>50% 算法异常)
   - 检查 IMU-LiDAR 外参标定
   - 检查时间同步
   - 调整算法参数

### 关键指标说明

- **IMU Preintegration Velocity Anomaly**: 最直接相关的异常，表明预积分计算出现问题
- **Position Jump**: 位置突变，可能由于定位失败或传感器数据异常
- **Sparse Cloud**: 点云过于稀疏，可能影响特征提取
- **Time Sync Issues**: 时间同步问题，可能导致数据融合错误

## 故障排除

1. **如果监控器没有检测到任何数据**
   - 检查 topic 名称是否正确（在 params.yaml 中配置）
   - 确保 ROS 节点都在运行
   - 检查 `rostopic list` 输出

2. **如果可视化脚本报错**
   - 确保安装了必要的 Python 包：`pip install pandas matplotlib numpy`
   - 检查 CSV 文件是否存在且格式正确

3. **如果分析结果不准确**
   - 调整 diagnostic_monitor.py 中的阈值参数
   - 确保数据采集时间足够长（至少运行完整个 bag 文件）

## 建议工作流程

1. 首先运行 `./run_diagnostic.sh monitor` 收集诊断数据
2. 查看生成的分析报告，了解问题的主要来源
3. 如需更详细的分析，使用可视化工具查看数据趋势
4. 根据诊断结果调整相应的参数或修复问题
5. 重新运行诊断，验证问题是否解决

## 注意事项

- 诊断系统会消耗一定的 CPU 资源，建议在测试环境中使用
- CSV 文件会持续增长，长时间运行请注意磁盘空间
- 可视化工具需要 GUI 环境，如在服务器上运行需要 X11 转发