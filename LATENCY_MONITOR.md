# LIO-SAM 延迟监控功能说明

## 已添加的延迟监控功能

### 1. 延迟数据采集

在 `realtime_plotter.py` 中已添加以下延迟监控功能：

#### 数据存储结构
```python
self.fusion_latency = deque(maxlen=50000)  # LIO-SAM 融合延迟
self.gps_latency = deque(maxlen=50000)     # GPS 处理延迟
```

#### 延迟计算方法
- **LIO-SAM 延迟**: `rospy.Time.now() - msg.header.stamp`
- **GPS 延迟**: `rospy.Time.now() - msg.header.stamp`
- 单位转换为毫秒（ms）

### 2. 新增的可视化图表（3x3 网格布局）

原有 6 个图表保持不变，新增 3 个延迟监控图表：

#### 第 7 个图表：LIO-SAM 处理延迟
- 位置：第三行第一列
- 内容：
  - 实时延迟曲线（蓝色）
  - 移动平均线（红色，窗口大小自适应）
  - GNSS 降级区域标记（橙色/绿色虚线）
  - Y轴：延迟（毫秒）
  - X轴：相对时间（秒）

#### 第 8 个图表：GPS 处理延迟
- 位置：第三行第二列
- 内容：
  - GPS 延迟曲线（绿色）
  - 移动平均线（红色）
  - 网格背景
  - 图例说明

#### 第 9 个图表：延迟统计信息
- 位置：第三行第三列
- 统计内容：
  - **LIO-SAM 延迟统计**：
    - 当前值
    - 平均值
    - 标准差
    - 最小/最大值
    - 95 百分位
  - **GPS 延迟统计**：
    - 当前值
    - 平均值
    - 标准差
    - 最小/最大值
  - **性能指标**：
    - ✓ 实时性能（<50ms）
    - ⚠ 近实时（50-100ms）
    - ✗ 高延迟（>100ms）

### 3. 数据保存功能

新增延迟数据 CSV 文件导出：
- `fusion_latency.csv` - LIO-SAM 延迟时间序列
- `gps_latency.csv` - GPS 延迟时间序列

### 4. 使用方法

#### 启动监控
```bash
# 运行 LIO-SAM 评估
roslaunch lio_sam run_evaluation.launch

# 延迟数据会自动采集并显示在 output/trajectory_realtime.png
```

#### 查看输出
- 实时图像：`./output/trajectory_realtime.png`（每 5 秒更新）
- 延迟数据：`./output/fusion_latency.csv`
- GPS 延迟：`./output/gps_latency.csv`

### 5. 延迟监控的意义

1. **实时性验证**：确认 LIO-SAM 是否满足实时要求
2. **性能瓶颈识别**：定位处理链中的延迟来源
3. **GNSS 降级影响**：评估 GNSS 信号丢失对延迟的影响
4. **系统优化**：为参数调优提供定量依据

### 6. 注意事项

- 延迟计算基于系统时钟，确保时钟同步
- 如果延迟为负值，可能存在时间同步问题
- 高延迟（>100ms）可能影响实时控制应用

### 7. 扩展建议

如需更详细的延迟分析，可以添加：
1. 点云预处理延迟（订阅 `/lio_sam/deskew/cloud_info`）
2. IMU 预积分延迟
3. 回环检测延迟
4. 各模块间的队列延迟

### 8. 延迟优化建议

如果观察到高延迟：
1. 检查 `lidarTimeOffset` 参数
2. 降低点云分辨率（增加 `downsampleRate`）
3. 减少特征提取数量
4. 优化 CPU 核心分配（`numberOfCores`）
5. 调整建图频率（`mappingProcessInterval`）