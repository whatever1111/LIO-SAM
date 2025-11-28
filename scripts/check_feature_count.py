#!/usr/bin/env python3
"""
Check feature extraction quality for LIO-SAM with Livox
"""

import rospy
from sensor_msgs.msg import PointCloud2
import struct

corner_counts = []
surface_counts = []
deskewed_counts = []

def count_points(msg):
    return msg.width * msg.height

def corner_callback(msg):
    count = count_points(msg)
    corner_counts.append(count)
    if len(corner_counts) % 10 == 0:
        avg = sum(corner_counts[-20:]) / min(20, len(corner_counts))
        print(f"Corner features: {count} (avg: {avg:.0f})")

def surface_callback(msg):
    count = count_points(msg)
    surface_counts.append(count)
    if len(surface_counts) % 10 == 0:
        avg = sum(surface_counts[-20:]) / min(20, len(surface_counts))
        print(f"Surface features: {count} (avg: {avg:.0f})")

def deskewed_callback(msg):
    count = count_points(msg)
    deskewed_counts.append(count)
    if len(deskewed_counts) % 10 == 0:
        avg = sum(deskewed_counts[-20:]) / min(20, len(deskewed_counts))
        print(f"Deskewed cloud: {count} (avg: {avg:.0f})")

def main():
    rospy.init_node('feature_count_checker')

    rospy.Subscriber('/lio_sam/feature/cloud_corner', PointCloud2, corner_callback)
    rospy.Subscriber('/lio_sam/feature/cloud_surface', PointCloud2, surface_callback)
    rospy.Subscriber('/lio_sam/deskew/cloud_deskewed', PointCloud2, deskewed_callback)

    print("Feature Count Monitor Started")
    print("=" * 50)

    rospy.spin()

if __name__ == '__main__':
    main()
