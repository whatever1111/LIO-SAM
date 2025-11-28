#!/usr/bin/env python3
"""
监控GPS和Fusion的XY位置关系
检测坐标系是否在某个时刻发生突变
同时监控CORRIMU的bias_comp和imu_status状态
"""

import rospy
import math
from nav_msgs.msg import Odometry
from fixposition_driver_msgs.msg import FpaImu

# IMU状态常量 (来自FpaConsts.msg)
IMU_STATUS_NAMES = {
    -1: "未指定",
    0: "未收敛",
    1: "热启动",
    2: "粗略收敛",
    3: "精细收敛"
}

class PositionMonitor:
    def __init__(self):
        rospy.init_node('position_monitor', anonymous=True)
        self.gps_pos = None
        self.fusion_pos = None
        self.count = 0

        # CORRIMU状态
        self.bias_comp = None
        self.imu_status = None
        self.imu_msg_count = 0

        rospy.Subscriber('/odometry/gps', Odometry, self.gps_cb)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.fusion_cb)
        rospy.Subscriber('/fixposition/fpa/corrimu', FpaImu, self.corrimu_cb)
        rospy.Timer(rospy.Duration(1.0), self.print_status)
        rospy.loginfo("位置监控启动 (含CORRIMU状态)")

    def corrimu_cb(self, msg):
        """解析CORRIMU消息中的bias_comp和imu_status"""
        self.bias_comp = msg.bias_comp
        self.imu_status = msg.imu_status
        self.imu_msg_count += 1

    def gps_cb(self, msg):
        self.gps_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def fusion_cb(self, msg):
        self.fusion_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def print_status(self, event):
        self.count += 1
        if self.gps_pos is None or self.fusion_pos is None:
            return

        gx, gy = self.gps_pos
        fx, fy = self.fusion_pos

        # 检查各种映射关系
        diff_direct = math.sqrt((fx-gx)**2 + (fy-gy)**2)  # GPS(x,y) -> Fusion(x,y)
        diff_swap = math.sqrt((fx-gy)**2 + (fy-gx)**2)    # GPS(x,y) -> Fusion(y,x)
        diff_negx = math.sqrt((fx+gx)**2 + (fy-gy)**2)    # GPS(x,y) -> Fusion(-x,y)
        diff_negy = math.sqrt((fx-gx)**2 + (fy+gy)**2)    # GPS(x,y) -> Fusion(x,-y)
        diff_swap_negx = math.sqrt((fx+gy)**2 + (fy-gx)**2)  # GPS(x,y) -> Fusion(-y,x)
        diff_swap_negy = math.sqrt((fx-gy)**2 + (fy+gx)**2)  # GPS(x,y) -> Fusion(y,-x)

        print(f"\n[{self.count:3d}s] GPS=({gx:7.2f}, {gy:7.2f}) | Fusion=({fx:7.2f}, {fy:7.2f})")
        print(f"      距离: 直接={diff_direct:.2f}m, 交换XY={diff_swap:.2f}m")
        print(f"      距离: -X={diff_negx:.2f}m, -Y={diff_negy:.2f}m")
        print(f"      距离: (-Y,X)={diff_swap_negx:.2f}m, (Y,-X)={diff_swap_negy:.2f}m")

        # 找出最佳匹配
        diffs = {
            '直接(x,y)': diff_direct,
            '交换(y,x)': diff_swap,
            '(-x,y)': diff_negx,
            '(x,-y)': diff_negy,
            '(-y,x)': diff_swap_negx,
            '(y,-x)': diff_swap_negy
        }
        best = min(diffs, key=diffs.get)
        print(f"      最佳匹配: {best} ({diffs[best]:.2f}m)")

        # 显示CORRIMU状态
        if self.bias_comp is not None:
            status_name = IMU_STATUS_NAMES.get(self.imu_status, f"未知({self.imu_status})")
            bias_str = "已补偿" if self.bias_comp else "未补偿"
            print(f"      CORRIMU: bias_comp={bias_str}, imu_status={status_name}, 消息数={self.imu_msg_count}")

def main():
    monitor = PositionMonitor()
    rospy.spin()

if __name__ == '__main__':
    main()
