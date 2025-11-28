#!/usr/bin/env python3
"""
实时航向角监控节点
订阅GPS和Fusion话题，实时对比航向角
"""

import rospy
import math
from nav_msgs.msg import Odometry
from collections import deque

class HeadingMonitor:
    def __init__(self):
        rospy.init_node('heading_monitor', anonymous=True)

        # 数据缓存
        self.gps_history = deque(maxlen=50)
        self.fusion_history = deque(maxlen=50)

        # 上一帧数据（用于计算速度）
        self.last_gps = None
        self.last_fusion = None

        # 统计
        self.gps_count = 0
        self.fusion_count = 0
        self.start_time = None

        # 订阅话题
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.fusion_callback)

        # 定时输出
        rospy.Timer(rospy.Duration(2.0), self.print_status)

        rospy.loginfo("航向角监控启动")
        rospy.loginfo("订阅: /odometry/gps, /lio_sam/mapping/odometry")

    def quat_to_yaw(self, qx, qy, qz, qw):
        """四元数转yaw角（度）"""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw)

    def velocity_to_heading(self, vx, vy):
        """速度转航向角（度）"""
        speed = math.sqrt(vx*vx + vy*vy)
        if speed < 0.3:
            return None
        return math.degrees(math.atan2(vy, vx))

    def normalize_angle(self, angle):
        """归一化到[-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def gps_callback(self, msg):
        self.gps_count += 1
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()

        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat_yaw = self.quat_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

        # 计算速度航向
        vel_heading = None
        if self.last_gps is not None:
            dt = t - self.last_gps['t']
            if dt > 0.001:
                vx = (x - self.last_gps['x']) / dt
                vy = (y - self.last_gps['y']) / dt
                vel_heading = self.velocity_to_heading(vx, vy)

        self.last_gps = {'t': t, 'x': x, 'y': y}
        self.gps_history.append({
            't': t,
            'x': x,
            'y': y,
            'quat_yaw': quat_yaw,
            'vel_heading': vel_heading
        })

    def fusion_callback(self, msg):
        self.fusion_count += 1
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()

        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quat_yaw = self.quat_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

        # 计算速度航向
        vel_heading = None
        if self.last_fusion is not None:
            dt = t - self.last_fusion['t']
            if dt > 0.001:
                vx = (x - self.last_fusion['x']) / dt
                vy = (y - self.last_fusion['y']) / dt
                vel_heading = self.velocity_to_heading(vx, vy)

        self.last_fusion = {'t': t, 'x': x, 'y': y}
        self.fusion_history.append({
            't': t,
            'x': x,
            'y': y,
            'quat_yaw': quat_yaw,
            'vel_heading': vel_heading
        })

    def print_status(self, event):
        if self.start_time is None:
            rospy.logwarn("等待数据...")
            return

        elapsed = rospy.Time.now().to_sec() - self.start_time

        print("\n" + "=" * 70)
        print(f"航向角分析 | 运行时间: {elapsed:.1f}s | GPS:{self.gps_count} Fusion:{self.fusion_count}")
        print("=" * 70)

        # 最新GPS数据
        if self.gps_history:
            g = self.gps_history[-1]
            print(f"GPS最新: pos=({g['x']:.2f}, {g['y']:.2f})")
            print(f"  四元数航向: {g['quat_yaw']:.1f}°")
            if g['vel_heading'] is not None:
                print(f"  速度航向:   {g['vel_heading']:.1f}°")

        # 最新Fusion数据
        if self.fusion_history:
            f = self.fusion_history[-1]
            print(f"Fusion最新: pos=({f['x']:.2f}, {f['y']:.2f})")
            print(f"  四元数航向: {f['quat_yaw']:.1f}°")
            if f['vel_heading'] is not None:
                print(f"  速度航向:   {f['vel_heading']:.1f}°")

        # 航向差异
        if self.gps_history and self.fusion_history:
            g = self.gps_history[-1]
            f = self.fusion_history[-1]

            quat_diff = self.normalize_angle(f['quat_yaw'] - g['quat_yaw'])
            print(f"\n航向差异（Fusion - GPS）:")
            print(f"  四元数航向差: {quat_diff:.1f}°")

            if g['vel_heading'] is not None and f['vel_heading'] is not None:
                vel_diff = self.normalize_angle(f['vel_heading'] - g['vel_heading'])
                print(f"  速度航向差:   {vel_diff:.1f}°")

            # 位置差异
            pos_diff = math.sqrt((f['x']-g['x'])**2 + (f['y']-g['y'])**2)
            print(f"  位置差异:     {pos_diff:.2f}m")

            # 诊断
            print("\n诊断:")
            if abs(quat_diff) < 15:
                print("  [OK] 四元数航向对齐良好")
            elif abs(quat_diff) > 75 and abs(quat_diff) < 105:
                print(f"  [WARN] 四元数航向差约90°，坐标系可能有问题")
            elif abs(quat_diff) > 165:
                print(f"  [WARN] 四元数航向差约180°，方向相反")
            else:
                print(f"  [INFO] 四元数航向差={quat_diff:.0f}°")

def main():
    try:
        monitor = HeadingMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
