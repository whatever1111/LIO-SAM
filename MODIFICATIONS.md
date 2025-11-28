# LIO-SAM 修改记录

## 修改日期: 2025-11-23

---

## 问题总结

### 原始问题
- **Fusion: 0** - 没有融合轨迹输出

### 根本原因
1. 点云时间戳单位错误（毫秒 vs 秒）
2. GPS 坐标系错误（ECEF vs ENU）
3. 特征提取参数不适合当前数据
4. 优化退化阈值过高

---

## 已完成的修改

### 1. 时间戳单位转换 ✅
**文件**: `src/imageProjection.cpp:239-240`

**原始代码**:
```cpp
dst.time = (src.timestamp - tmpCustomCloudIn->points[0].timestamp);
```

**修改后**:
```cpp
// timestamp is in milliseconds, convert to seconds
dst.time = (src.timestamp - tmpCustomCloudIn->points[0].timestamp) / 1000.0;
```

**原因**: 点云的 timestamp 字段是毫秒单位，但代码按秒处理，导致扫描时间变成 100 秒而不是 0.1 秒。

---

### 2. ECEF 到 ENU 坐标转换 ✅
**文件**: `src/fpaOdomConverter.cpp`

**主要修改**:
- 添加 `Eigen/Dense` 和 `cmath` 头文件
- 添加成员变量: `origin_initialized`, `origin_ecef`, `R_ecef_to_enu`
- 添加函数 `initializeOrigin()`: 计算第一个 GPS 位置的经纬度，构建 ECEF→ENU 旋转矩阵
- 添加函数 `ecefToEnu()`: 将 ECEF 坐标转换为相对于原点的 ENU 坐标
- 修改 `fpaOdomCallback()`: 调用坐标转换

**原因**: GPS 输出的是 ECEF（地心地固）坐标（百万米级别），而 LIO-SAM 期望局部坐标系，导致巨大的位置跳变。

---

### 3. 特征提取参数调整 ✅
**文件**: `config/params.yaml:56-60`

**原始配置**:
```yaml
edgeThreshold: 1.0
surfThreshold: 0.1
edgeFeatureMinValidNum: 10
surfFeatureMinValidNum: 100
```

**修改后**:
```yaml
edgeThreshold: 0.1                              # 降低以提取更多 edge 特征
surfThreshold: 0.1
edgeFeatureMinValidNum: 5                       # 降低以适应稀疏场景
surfFeatureMinValidNum: 50                      # 降低以适应稀疏场景
```

**原因**: 原始阈值太严格，导致 edge 特征太少（只有 10 个），优化不稳定。

---

### 3.1 传感器参数修正 ✅ (关键修复!)
**文件**: `config/params.yaml:26-28`

**原始配置**:
```yaml
N_SCAN: 128
Horizon_SCAN: 1800
```

**修改后**:
```yaml
N_SCAN: 16                    # 实际数据是16线 (0-15)
Horizon_SCAN: 5000            # 每线约4992-5016点
```

**原因**: 实际LiDAR数据是16线而非128线。错误的配置导致:
- Range image投影错误
- 特征提取索引错误
- 点云去畸变失败

---

### 4. 优化退化阈值调整 ✅
**文件**: `src/mapOptmization.cpp:1244`

**原始代码**:
```cpp
float eignThre[6] = {100, 100, 100, 100, 100, 100};
```

**修改后**:
```cpp
float eignThre[6] = {10, 10, 10, 10, 10, 10};  // Lowered from 100
```

**原因**: 特征值 `[97.4, 42.6, 24.3]` 小于阈值 100，导致所有帧都被标记为退化。

---

### 4.1 第一帧位置先验收紧 ✅
**文件**: `src/mapOptmization.cpp:1404-1406`

**原始代码**:
```cpp
priorNoise = (Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8)
```

**修改后**:
```cpp
priorNoise = (Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e-4, 1e-4, 1e-4)
```

**原因**: 原始 X,Y,Z 方差 1e8 m² 导致第一帧位置几乎无约束，造成初始位置漂移和 Large velocity 警告。

---

### 4.2 GPS 激活距离降低 ✅
**文件**: `src/mapOptmization.cpp:1428-1430`

**原始代码**:
```cpp
if (pointDistance(...) < 5.0)
    return;
```

**修改后**:
```cpp
if (pointDistance(...) < 1.0)
    return;
```

**原因**: 原始设置需要移动5米才启用GPS，导致初始帧完全没有GPS约束。降低到1米使GPS更早介入稳定系统。

---

### 4.3 IMU队列大小限制 ✅
**文件**: `src/imuPreintegration.cpp:467-472`

**新增代码**:
```cpp
// Limit queue size to prevent memory issues (keep ~10 seconds of data at 200Hz)
const size_t maxQueueSize = 2000;
while (imuQueOpt.size() > maxQueueSize)
    imuQueOpt.pop_front();
while (imuQueImu.size() > maxQueueSize)
    imuQueImu.pop_front();
```

**原因**: 如果没有odometry消息来消费IMU数据，队列会无限增长导致std::bad_alloc内存崩溃。

---

### 5. 诊断日志添加 ✅
**文件**: `src/mapOptmization.cpp`

**添加位置**:
- `extractSurroundingKeyFrames()`: 打印关键帧数量
- `scan2MapOptimization()`: 打印无关键帧警告
- 特征值检查处: 打印退化时的特征值

**文件**: `src/imageProjection.cpp`

**添加位置**:
- `deskewInfo()`: 打印 IMU 时间戳不匹配详情

---

### 6. Python 脚本修复 ✅
**文件**: `scripts/record_trajectory.py`

**修改内容**:
- 添加 `signal` 模块导入
- 回调函数移到变量初始化之后
- 添加信号处理器确保优雅关闭
- 添加 `bag_ready` 延迟标志 (0.5秒)
- 使用 `rospy.logwarn_throttle` 报告写入错误 (每5秒限流)

**重要**: 错误应该被报告而不是静默忽略!

---

### 7. 评估脚本修复 ✅
**文件**: `scripts/evaluate_trajectory.py:32-33`

**修改后**:
```python
# Allow reading unindexed bags (in case of improper shutdown)
bag = rosbag.Bag(self.bag_file, 'r', allow_unindexed=True)
```

---

### 8. 运行脚本修复 ✅
**文件**: `run_evaluation.sh`

**修改内容**:
- 使用 `kill -2` (SIGINT) 替代 `kill` (SIGTERM)
- 添加 `wait` 命令等待进程正确退出

---

## 待解决问题

### 1. 优化退化阈值需要重新编译
当前特征值 `[97.4, 42.6, 24.3]` > 阈值 10，但日志仍显示退化，说明编译未完成。

### 2. IMU 时间戳不匹配
```
Cloud time: [1763444695.982, 1763444696.083]
IMU time:   [1763444696.085, 1763444696.305]
```
IMU 数据比点云晚约 0.1 秒，导致前几帧被跳过。

### 3. Large velocity, reset IMU-preintegration
频繁的速度重置，需要进一步调查原因。

### 4. Failed to write degraded: 0
`/gnss_degraded` 话题的 bag 写入失败，可能是消息类型问题。

---

## 重新编译命令

```bash
cd /root/autodl-tmp/catkin_ws
catkin_make -j4
```

## 测试命令

```bash
cd /root/autodl-tmp/catkin_ws
bash src/LIO-SAM/run_evaluation.sh
```

---

## 数据信息

- **Bag 文件**: `/root/autodl-tmp/info_fixed.bag`
- **时长**: 14:40 (880秒)
- **GPS 原点**:
  - ECEF: [-2426035.96, 4254995.00, 4071649.55]
  - LLA: 39.925513°N, 119.690140°E
- **点云**: `/lidar_points`, 8802 帧, 10Hz
- **IMU**: `/imu/data`, 176023 帧, 200Hz
