#!/bin/bash

# LIO-SAM 参数调整建议脚本
# 基于 bag 文件验证结果的参数更新

echo "======================================"
echo "LIO-SAM 参数配置更新建议"
echo "基于 info_fixed.bag 验证结果"
echo "======================================"

# 备份原配置文件
CONFIG_FILE="/root/autodl-tmp/catkin_ws/src/LIO-SAM/config/params.yaml"
BACKUP_FILE="/root/autodl-tmp/catkin_ws/src/LIO-SAM/config/params.yaml.backup.$(date +%Y%m%d_%H%M%S)"

echo "1. 备份原配置文件..."
cp $CONFIG_FILE $BACKUP_FILE
echo "   备份保存至: $BACKUP_FILE"

echo ""
echo "2. 建议的参数修改:"
echo ""
echo "   ❌ 必须修改:"
echo "   - lidarTimeOffset: 0.097 → 0.212 (基于实测时间偏移)"
echo ""
echo "   ⚠️ 可选优化:"
echo "   - lidarMinRange: 1.0 → 0.5 (包含更多近距离点)"
echo ""
echo "   📝 确认正确的参数:"
echo "   - sensor: livox ✓"
echo "   - N_SCAN: 16 ✓"
echo "   - Horizon_SCAN: 5000 ✓"
echo "   - pointCloudTopic: /lidar_points ✓"
echo "   - imuTopic: /imu/data ✓"
echo ""
echo "3. GPS 数据处理:"
echo "   需要运行 FPA Odometry Converter 将 FPA 数据转换为 /odometry/gps"
echo "   原始 FPA 话题已确认存在:"
echo "   - /fixposition/fpa/llh (9.76 Hz)"
echo "   - /fixposition/fpa/odometry (9.75 Hz)"
echo "   - /fixposition/fpa/odomenu (9.76 Hz)"
echo ""

# 询问是否应用修改
echo "======================================"
echo "是否应用建议的参数修改？"
echo "这将修改 lidarTimeOffset: 0.097 → 0.212"
echo "原文件已备份至: $BACKUP_FILE"
echo ""
echo "如需手动修改，请编辑:"
echo "$CONFIG_FILE"
echo "第32行: lidarTimeOffset: 0.212"
echo "======================================"

# 显示需要检查的内容
echo ""
echo "4. 后续步骤:"
echo "   a) 检查并启动 FPA Odometry Converter 节点"
echo "   b) 验证 /odometry/gps 话题生成"
echo "   c) 使用更新的参数运行 LIO-SAM"
echo "   d) 监控时间同步质量"
echo ""
echo "验证报告已保存至:"
echo "/root/autodl-tmp/catkin_ws/src/LIO-SAM/VERIFICATION_REPORT.md"