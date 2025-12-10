#!/usr/bin/env python3
"""
Republish /lidar_points with frame_id changed from lidar_link to base_link
"""
import rospy
from sensor_msgs.msg import PointCloud2

pub = None

def callback(msg):
    msg.header.frame_id = "base_link"
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('lidar_frame_republisher')
    pub = rospy.Publisher('/lidar_points_baselink', PointCloud2, queue_size=1)
    rospy.Subscriber('/lidar_points', PointCloud2, callback)
    rospy.loginfo("Republishing /lidar_points -> /lidar_points_baselink with frame_id=base_link")
    rospy.spin()
