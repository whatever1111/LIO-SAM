#!/usr/bin/env python3
"""
检查GPS内部一致性：四元数航向 vs 速度航向
"""

import rospy
import math
from nav_msgs.msg import Odometry

class GPSConsistencyChecker:
    def __init__(self):
        rospy.init_node('gps_consistency_checker', anonymous=True)
        self.last_pos = None
        self.last_time = None
        self.count = 0

        rospy.Subscriber('/odometry/gps', Odometry, self.callback)
        rospy.loginfo("检查GPS四元数和速度方向的一致性")

    def callback(self, msg):
        self.count += 1
        t = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 四元数航向
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        quat_yaw = math.degrees(math.atan2(siny, cosy))

        # 速度航向（从位置差分）
        vel_yaw = None
        if self.last_pos is not None:
            dt = t - self.last_time
            if dt > 0.01:
                vx = (x - self.last_pos[0]) / dt
                vy = (y - self.last_pos[1]) / dt
                speed = math.sqrt(vx*vx + vy*vy)
                if speed > 0.5:
                    vel_yaw = math.degrees(math.atan2(vy, vx))

        self.last_pos = (x, y)
        self.last_time = t

        if self.count % 20 == 0:
            if vel_yaw is not None:
                diff = vel_yaw - quat_yaw
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                print(f"GPS #{self.count}: 四元数={quat_yaw:.1f}°, 速度={vel_yaw:.1f}°, 差={diff:.1f}°")
            else:
                print(f"GPS #{self.count}: 四元数={quat_yaw:.1f}°, 速度=N/A (速度太小)")

def main():
    checker = GPSConsistencyChecker()
    rospy.spin()

if __name__ == '__main__':
    main()
