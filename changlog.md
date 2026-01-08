# Changelog (LIO-SAM fork)

本文件记录本仓库相对上游 LIO-SAM 的主要行为改动，重点覆盖 GNSS/Pose3 因子与退化(Degrade/Degenerate)处理链路。

## 2025-12-31

### Added
- **GNSS Pose3(6DoF) Prior 因子（可选）**：在 `src/mapOptmization.cpp:addGPSFactor()` 中支持从 `/odometry/gps` 读取 **完整 6×6 pose 协方差**，并进行：
  - ENU→LiDAR 的 6×6 协方差旋转
  - ROS `[pos, ori]` → GTSAM Pose3 tangent `[ori, pos]` 的重排
  - 协方差方差下限（`gpsPosStdFloor`, `gpsOriStdFloorDeg`）
  - SPD 投影（避免协方差非正定导致 ISAM2 崩溃）
  - 鲁棒核包装（`gpsRobustKernel`, `gpsRobustDelta`）
- `include/covariance_utils.h`：提供协方差旋转/重排/SPD 投影/非有限值清洗等纯数学工具函数。
- `msg/MappingStatus.msg`：新增 mapping 状态消息，显式发布 `is_degenerate`。
- `src/tfPublisher.cpp`：新增 TF Publisher（从 `params.yaml` 发布静态 TF），替代部分 launch 里硬编码的 static_transform_publisher。

### Changed
- **退化标志通道改造**：
  - `src/mapOptmization.cpp` 不再把退化标志写入 `odometry_incremental.pose.covariance[0]`，改为发布：
    - `/lio_sam/mapping/odometry_incremental`（保持 pose 输出，但不再复用协方差字段表达“退化标志”）
    - `/lio_sam/mapping/odometry_incremental_status`（`lio_sam/MappingStatus`，字段 `is_degenerate`）
  - `src/imuPreintegration.cpp` 使用 `message_filters` 对 incremental odom 与 status 进行时间同步，保证 IMU 图里使用正确的退化标志（并相应切换 correctionNoise / correctionNoise2）。
  - `src/imageProjection.cpp` 在 deskew 时改为检查 `MappingStatus`（避免同一扫描周期内退化状态翻转导致的 deskew 不一致），替代上游的 `covariance[0]` hack。
- `config/params.yaml` / `include/utility.h`：新增 GNSS 相关参数（方差下限、鲁棒核、yaw-only、GNSS degraded gating 等）以及 IMU correction noise 参数。

### Fixed
- 修复/规避 GNSS 协方差 NaN/Inf、非对称、非 SPD 等导致的 `noiseModel::Gaussian::Covariance(...)` / ISAM2 数值异常问题（通过清洗与 SPD 投影）。

### Tooling / Scripts
- 多个分析脚本改为订阅 `/lio_sam/mapping/odometry_incremental_status` 获取退化标志（替代解析 `pose.covariance[0]`）。

### Tests
- 新增 gtest：`test/test_covariance_utils.cpp`，覆盖 rotate/reorder/SPD/sanitize 等核心数学逻辑，避免后续改动引入协方差处理回归。

