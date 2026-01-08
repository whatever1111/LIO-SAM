#!/usr/bin/env python3
"""
Republish a PointCloud2 topic into a target frame using TF (tf2).

Use case (REP-103 fix for this dataset):
  - Bag publishes /lidar_points with frame_id = 'lidar_link' but its +X appears to point backward.
  - Publish a static TF: lidar_link -> lidar_link_rep103 (Rz(pi))
  - Transform the cloud into lidar_link_rep103 and publish /lidar_points_rep103

This keeps the "fix" explicit in TF and produces a corrected pointcloud for consumers
that do not apply TF themselves (e.g., LIO-SAM).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from tf.transformations import quaternion_from_euler

try:
    from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud as _do_transform_cloud  # type: ignore

    _HAS_TF2_SENSOR_MSGS = True
except Exception:
    _HAS_TF2_SENSOR_MSGS = False
    _do_transform_cloud = None


@dataclass(frozen=True)
class StaticTfSpec:
    parent: str
    child: str
    x: float
    y: float
    z: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def _publish_static_tf(spec: StaticTfSpec) -> None:
    br = tf2_ros.StaticTransformBroadcaster()
    tf_msg = TransformStamped()
    tf_msg.header.stamp = rospy.Time.now()
    tf_msg.header.frame_id = spec.parent
    tf_msg.child_frame_id = spec.child
    tf_msg.transform.translation.x = spec.x
    tf_msg.transform.translation.y = spec.y
    tf_msg.transform.translation.z = spec.z
    qx, qy, qz, qw = quaternion_from_euler(
        math.radians(spec.roll_deg),
        math.radians(spec.pitch_deg),
        math.radians(spec.yaw_deg),
    )
    tf_msg.transform.rotation.x = float(qx)
    tf_msg.transform.rotation.y = float(qy)
    tf_msg.transform.rotation.z = float(qz)
    tf_msg.transform.rotation.w = float(qw)
    br.sendTransform(tf_msg)


def _quat_to_rotmat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # ROS quaternion order: (x, y, z, w)
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _numpy_dtype_for_pointfield(field: PointField, is_bigendian: bool) -> np.dtype:
    endian = ">" if is_bigendian else "<"
    mapping = {
        PointField.INT8: "i1",
        PointField.UINT8: "u1",
        PointField.INT16: "i2",
        PointField.UINT16: "u2",
        PointField.INT32: "i4",
        PointField.UINT32: "u4",
        PointField.FLOAT32: "f4",
        PointField.FLOAT64: "f8",
    }
    code = mapping.get(int(field.datatype))
    if code is None:
        raise ValueError(f"Unsupported PointField datatype: {field.datatype}")
    base = np.dtype(endian + code)
    if int(field.count) == 1:
        return base
    return np.dtype((base, (int(field.count),)))


def _transform_cloud_manual(cloud: PointCloud2, tf_msg: TransformStamped) -> PointCloud2:
    """
    Manual PointCloud2 transform fallback when tf2_sensor_msgs is unavailable.
    Only transforms x/y/z; other fields are preserved bitwise.
    """
    fields = cloud.fields
    names = [f.name for f in fields]
    if "x" not in names or "y" not in names or "z" not in names:
        raise ValueError("PointCloud2 missing x/y/z fields")

    dtype = np.dtype(
        {
            "names": names,
            "formats": [_numpy_dtype_for_pointfield(f, cloud.is_bigendian) for f in fields],
            "offsets": [int(f.offset) for f in fields],
            "itemsize": int(cloud.point_step),
        }
    )
    arr = np.frombuffer(cloud.data, dtype=dtype)
    out_arr = np.array(arr, copy=True)

    R = _quat_to_rotmat_xyzw(
        tf_msg.transform.rotation.x,
        tf_msg.transform.rotation.y,
        tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w,
    )
    t = np.array(
        [
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ],
        dtype=np.float32,
    )

    xyz = np.stack([out_arr["x"].astype(np.float32), out_arr["y"].astype(np.float32), out_arr["z"].astype(np.float32)], axis=1)
    xyz_t = (xyz @ R.T) + t[None, :]
    out_arr["x"] = xyz_t[:, 0]
    out_arr["y"] = xyz_t[:, 1]
    out_arr["z"] = xyz_t[:, 2]

    cloud_out = PointCloud2()
    cloud_out.header = cloud.header
    cloud_out.header.frame_id = tf_msg.header.frame_id  # target frame
    cloud_out.height = cloud.height
    cloud_out.width = cloud.width
    cloud_out.fields = cloud.fields
    cloud_out.is_bigendian = cloud.is_bigendian
    cloud_out.point_step = cloud.point_step
    cloud_out.row_step = cloud.row_step
    cloud_out.is_dense = cloud.is_dense
    cloud_out.data = out_arr.tobytes()
    return cloud_out

class LidarTfRepublisher:
    def __init__(self) -> None:
        self.input_topic = rospy.get_param("~input_topic", "/lidar_points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_points_rep103")
        self.target_frame = rospy.get_param("~target_frame", "lidar_link_rep103")
        self.source_frame_override: Optional[str] = rospy.get_param("~source_frame", None)

        self.publish_static_tf = bool(rospy.get_param("~publish_static_tf", True))
        self.static_parent = rospy.get_param("~static_parent_frame", "lidar_link")
        self.static_child = rospy.get_param("~static_child_frame", self.target_frame)
        self.static_xyz = rospy.get_param("~static_xyz", [0.0, 0.0, 0.0])
        self.static_rpy_deg = rospy.get_param("~static_rpy_deg", [0.0, 0.0, 180.0])

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self._cb, queue_size=1)

        if self.publish_static_tf:
            if len(self.static_xyz) != 3 or len(self.static_rpy_deg) != 3:
                raise rospy.ROSException("~static_xyz and ~static_rpy_deg must have 3 elements each")
            _publish_static_tf(
                StaticTfSpec(
                    parent=str(self.static_parent),
                    child=str(self.static_child),
                    x=float(self.static_xyz[0]),
                    y=float(self.static_xyz[1]),
                    z=float(self.static_xyz[2]),
                    roll_deg=float(self.static_rpy_deg[0]),
                    pitch_deg=float(self.static_rpy_deg[1]),
                    yaw_deg=float(self.static_rpy_deg[2]),
                )
            )
            rospy.loginfo("Published static TF: %s -> %s rpy(deg)=%s xyz=%s",
                          self.static_parent, self.static_child, self.static_rpy_deg, self.static_xyz)

        rospy.loginfo("republish_lidar_tf: %s (%s) -> %s (%s)",
                      self.input_topic,
                      self.source_frame_override if self.source_frame_override else "<msg.frame_id>",
                      self.output_topic,
                      self.target_frame)

    def _cb(self, msg: PointCloud2) -> None:
        src = self.source_frame_override if self.source_frame_override else msg.header.frame_id
        if not src:
            rospy.logwarn_throttle(5.0, "PointCloud2 has empty frame_id; skipping")
            return

        cloud_in = msg
        if self.source_frame_override and cloud_in.header.frame_id != self.source_frame_override:
            cloud_in = PointCloud2()
            cloud_in.header = msg.header
            cloud_in.height = msg.height
            cloud_in.width = msg.width
            cloud_in.fields = msg.fields
            cloud_in.is_bigendian = msg.is_bigendian
            cloud_in.point_step = msg.point_step
            cloud_in.row_step = msg.row_step
            cloud_in.data = msg.data
            cloud_in.is_dense = msg.is_dense
            cloud_in.header.frame_id = self.source_frame_override

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                src,
                cloud_in.header.stamp,
                timeout=rospy.Duration(0.05),
            )
        except Exception as exc:
            # Static TF may be latched at time=0; fall back to latest.
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    src,
                    rospy.Time(0),
                    timeout=rospy.Duration(0.05),
                )
            except Exception:
                rospy.logwarn_throttle(2.0, "TF lookup failed (%s -> %s): %s", src, self.target_frame, str(exc))
                return

        try:
            if _HAS_TF2_SENSOR_MSGS and _do_transform_cloud is not None:
                cloud_out = _do_transform_cloud(cloud_in, tf_msg)
            else:
                cloud_out = _transform_cloud_manual(cloud_in, tf_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "do_transform_cloud failed: %s", str(exc))
            return

        self.pub.publish(cloud_out)


def main() -> None:
    rospy.init_node("republish_lidar_tf", anonymous=True)
    _ = LidarTfRepublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
