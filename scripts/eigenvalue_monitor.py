#!/usr/bin/env python3
"""
监控LMOptimization的特征值，诊断退化检测问题
需要在mapOptmization.cpp中添加特征值发布功能
"""

import rospy
from std_msgs.msg import Float64MultiArray
import numpy as np

class EigenvalueMonitor:
    def __init__(self):
        self.eigenvalues = []

        # 当前退化阈值
        self.thresholds = [100, 100, 100, 100, 100, 100]

        rospy.loginfo("="*60)
        rospy.loginfo("特征值监控器")
        rospy.loginfo("="*60)
        rospy.loginfo("当前退化阈值: %s", self.thresholds)
        rospy.loginfo("")
        rospy.loginfo("如果特征值持续低于阈值，需要:")
        rospy.loginfo("  1. 降低退化阈值")
        rospy.loginfo("  2. 或提高特征点质量")
        rospy.loginfo("="*60)

        # 订阅特征值 (需要在代码中添加发布)
        rospy.Subscriber('/lio_sam/mapping/eigenvalues',
                        Float64MultiArray, self.eigen_cb)

    def eigen_cb(self, msg):
        eig = np.array(msg.data)
        self.eigenvalues.append(eig)

        # 检查退化
        degenerate_dims = []
        for i in range(6):
            if eig[i] < self.thresholds[i]:
                degenerate_dims.append(i)

        if degenerate_dims:
            rospy.logwarn("退化维度: %s", degenerate_dims)
            rospy.logwarn("特征值: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]",
                         eig[0], eig[1], eig[2], eig[3], eig[4], eig[5])

def main():
    rospy.init_node('eigenvalue_monitor')
    monitor = EigenvalueMonitor()

    print("""
注意: 此脚本需要在mapOptmization.cpp中添加特征值发布功能。

在LMOptimization函数中，cv::eigen(matAtA, matE, matV)之后添加:

// 发布特征值用于调试
std_msgs::Float64MultiArray eigenMsg;
for (int i = 0; i < 6; i++) {
    eigenMsg.data.push_back(matE.at<float>(0, i));
}
pubEigenvalues.publish(eigenMsg);

并在构造函数中添加:
pubEigenvalues = nh.advertise<std_msgs::Float64MultiArray>("lio_sam/mapping/eigenvalues", 1);

当前建议的解决方案:
1. 降低退化阈值从100到10或更低
2. 在mapOptmization.cpp第1239行修改:
   float eignThre[6] = {10, 10, 10, 10, 10, 10};  // 从100降到10
""")

    rospy.spin()

if __name__ == '__main__':
    main()
