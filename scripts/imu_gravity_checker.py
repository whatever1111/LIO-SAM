#!/usr/bin/env python3
"""
直接检查IMU数据经过坐标变换后的重力方向
这是定位Y方向漂移的关键
"""

import rospy
import numpy as np
from fixposition_driver_msgs.msg import FpaImu
from sensor_msgs.msg import Imu
from collections import deque

class IMUGravityChecker:
    def __init__(self):
        # 外参旋转矩阵 (从params.yaml)
        # extrinsicRot: [-1, 0, 0, 0, -1, 0, 0, 0, 1]
        self.extRot = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])

        self.raw_acc_samples = deque(maxlen=200)
        self.transformed_acc_samples = deque(maxlen=200)
        self.sample_count = 0

        # 订阅原始IMU数据
        rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.fpa_imu_cb)

        rospy.loginfo("="*70)
        rospy.loginfo("IMU重力方向检查器")
        rospy.loginfo("="*70)
        rospy.loginfo("外参旋转矩阵 (extrinsicRot):")
        rospy.loginfo("  %s", self.extRot[0])
        rospy.loginfo("  %s", self.extRot[1])
        rospy.loginfo("  %s", self.extRot[2])
        rospy.loginfo("")
        rospy.loginfo("GTSAM使用MakeSharedU(g)，假设重力在+Z方向")
        rospy.loginfo("如果变换后重力不在Z轴，就会产生错误的速度积分！")
        rospy.loginfo("="*70)

    def fpa_imu_cb(self, msg):
        self.sample_count += 1

        # 原始加速度
        raw_acc = np.array([
            msg.data.linear_acceleration.x,
            msg.data.linear_acceleration.y,
            msg.data.linear_acceleration.z
        ])

        # 变换后的加速度 (与imuConverter相同的变换)
        transformed_acc = self.extRot @ raw_acc

        self.raw_acc_samples.append(raw_acc)
        self.transformed_acc_samples.append(transformed_acc)

        # 每100个样本报告一次
        if self.sample_count % 100 == 0:
            self.report()

    def report(self):
        if len(self.transformed_acc_samples) < 10:
            return

        raw_arr = np.array(list(self.raw_acc_samples))
        trans_arr = np.array(list(self.transformed_acc_samples))

        raw_mean = np.mean(raw_arr, axis=0)
        trans_mean = np.mean(trans_arr, axis=0)

        rospy.loginfo("-"*70)
        rospy.loginfo("样本数: %d", self.sample_count)
        rospy.loginfo("")
        rospy.loginfo("原始IMU加速度 (平均):")
        rospy.loginfo("  [%.4f, %.4f, %.4f] m/s²", raw_mean[0], raw_mean[1], raw_mean[2])
        rospy.loginfo("  范数: %.4f (期望≈9.81)", np.linalg.norm(raw_mean))
        rospy.loginfo("")
        rospy.loginfo("变换后加速度 (平均):")
        rospy.loginfo("  [%.4f, %.4f, %.4f] m/s²", trans_mean[0], trans_mean[1], trans_mean[2])
        rospy.loginfo("  范数: %.4f", np.linalg.norm(trans_mean))

        # 分析重力分量
        rospy.loginfo("")
        rospy.loginfo(">>> 重力分量分析:")

        # GTSAM的MakeSharedU(g)假设重力在+Z方向，即静止时acc_z ≈ +g
        # 如果不是，重力补偿就会出错

        x_component = abs(trans_mean[0])
        y_component = abs(trans_mean[1])
        z_component = abs(trans_mean[2])

        rospy.loginfo("  X分量: %.4f m/s² (%.1f%%)", trans_mean[0], 100*x_component/9.81)
        rospy.loginfo("  Y分量: %.4f m/s² (%.1f%%)", trans_mean[1], 100*y_component/9.81)
        rospy.loginfo("  Z分量: %.4f m/s² (%.1f%%)", trans_mean[2], 100*z_component/9.81)

        # 判断是否有问题
        if y_component > 0.5:  # Y方向有超过0.5 m/s²的分量
            rospy.logerr("")
            rospy.logerr("!!! 警告: Y方向有显著加速度分量 (%.2f m/s²) !!!", trans_mean[1])
            rospy.logerr("    GTSAM认为这是线性加速度，会积分成Y方向速度！")
            rospy.logerr("    这就是Y方向漂移的根本原因！")
            rospy.logerr("")
            rospy.logerr("    可能的原因:")
            rospy.logerr("    1. IMU安装有倾斜角度")
            rospy.logerr("    2. extrinsicRot配置与实际IMU方向不匹配")
            rospy.logerr("    3. 车辆不是水平的")

        if x_component > 0.5:
            rospy.logwarn("  X方向也有分量: %.2f m/s²", trans_mean[0])

        # 检查Z分量是否正确
        if trans_mean[2] < 8:
            rospy.logwarn("  Z分量偏小 (%.2f)，应该接近9.81", trans_mean[2])
        elif trans_mean[2] < 0:
            rospy.logerr("  Z分量为负! 需要使用MakeSharedD()而不是MakeSharedU()")

def main():
    rospy.init_node('imu_gravity_checker')
    checker = IMUGravityChecker()

    rospy.loginfo("开始监控IMU数据... (Ctrl+C 停止)")
    rospy.loginfo("请保持车辆静止以获得准确的重力测量")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
