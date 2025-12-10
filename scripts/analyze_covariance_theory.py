#!/usr/bin/env python3
"""
分析LIO-SAM协方差计算原理及其与GPS融合的关系
"""

print("""
================================================================================
              LIO-SAM 协方差计算原理分析
================================================================================

## 1. 协方差的来源

协方差来自GTSAM的iSAM2增量式平滑与建图算法：

```cpp
poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
```

这是当前最新位姿的**边缘协方差矩阵** (6x6)，表示位姿估计的不确定性。

矩阵结构：
- poseCovariance(0,0), (1,1), (2,2): roll, pitch, yaw的方差 (rad²)
- poseCovariance(3,3), (4,4), (5,5): x, y, z的方差 (m²)

GPS融合检查的是:
- poseCovariance(3,3): X位置方差
- poseCovariance(4,4): Y位置方差

## 2. 协方差如何被计算

### 2.1 因子图结构

LIO-SAM的因子图包含：

1. **先验因子** (仅第一帧):
   ```cpp
   priorNoise = Variances([1e-2, 1e-2, π², 1e8, 1e8, 1e8])
   //           roll   pitch  yaw   x    y    z
   ```
   初始位置方差非常大(1e8 m²)，表示初始位置完全未知

2. **里程计因子** (帧间):
   ```cpp
   odometryNoise = Variances([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
   //              roll   pitch  yaw    x     y     z
   ```
   帧间噪声很小: 旋转0.001rad, 平移0.01m

3. **GPS因子** (当添加时):
   ```cpp
   gps_noise = Variances([noise_x*scale, noise_y*scale, noise_z])
   ```
   GPS噪声由GPS传感器报告的协方差决定

4. **回环因子** (当检测到时)

### 2.2 协方差传播机制

ISAM2使用贝叶斯树进行增量式优化。协方差计算：

P(x_n) = (J^T * Σ^{-1} * J)^{-1}

其中：
- J: 雅可比矩阵
- Σ: 测量噪声协方差

**关键理解**:
- 每添加一个里程计因子，协方差会**增长**（不确定性累积）
- 每添加一个GPS/回环因子，协方差会**减小**（全局约束）

## 3. 为什么380秒后协方差很低？

### 3.1 理论分析

从日志看到：
```
poseCov=[0.212, 0.389] < threshold=25.0
poseCov=[0.150, 1.162] < threshold=25.0
```

协方差始终在0.1-15范围内，远低于25的阈值。

### 3.2 原因分析

**原因1: 里程计噪声设置太小**
```cpp
odometryNoise = Variances([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])
```
- 位置噪声: 1e-4 m² = 0.01m标准差
- 这意味着LIO-SAM认为每帧定位误差只有1cm

**原因2: LiDAR-IMU配准很稳定**
- 在特征丰富区域，ICP/NDT配准给出稳定结果
- 配准残差小 → 噪声估计小 → 协方差不增长

**原因3: 协方差累积被稀释**
- 因子图包含数千个节点
- 单个因子的不确定性被分散到整个图中
- 边缘协方差反映的是**局部**不确定性，不是全局漂移

### 3.3 核心问题

**协方差只能反映局部一致性，无法检测全局漂移！**

想象一条直线：
```
真实轨迹:   A -------- B -------- C
估计轨迹:   A -------- B' ------- C'
                        ↑
                   局部平滑但全局偏移
```

每一段AB、BC都很平滑（低协方差），但整体有漂移。

## 4. 当前是否输出协方差？

从代码看，协方差**没有被直接输出到文件**：
- 原来的cout输出被注释掉了:
  ```cpp
  // cout << "Pose covariance:" << endl;
  // cout << isam->marginalCovariance(...) << endl;
  ```
- 只在GPS跳过时输出警告日志

## 5. 建议的诊断方案

为了验证上述分析，建议添加协方差输出功能。
""")

# 生成修改建议
print("""
================================================================================
                    协方差诊断修改建议
================================================================================

在 mapOptmization.cpp 的 saveKeyFramesAndFactor() 函数末尾添加：

```cpp
// 在 poseCovariance = isam->marginalCovariance(...) 之后添加:

// 输出协方差到CSV文件（用于诊断）
static std::ofstream covFile("/root/autodl-tmp/catkin_ws/src/LIO-SAM/output/pose_covariance.csv");
static bool headerWritten = false;
if (!headerWritten) {
    covFile << "time,cov_roll,cov_pitch,cov_yaw,cov_x,cov_y,cov_z" << std::endl;
    headerWritten = true;
}
covFile << std::fixed << std::setprecision(6)
        << timeLaserInfoCur << ","
        << poseCovariance(0,0) << ","
        << poseCovariance(1,1) << ","
        << poseCovariance(2,2) << ","
        << poseCovariance(3,3) << ","
        << poseCovariance(4,4) << ","
        << poseCovariance(5,5) << std::endl;
```

这样可以记录每个关键帧的协方差，用于后续分析。
""")
