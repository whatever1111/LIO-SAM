#!/usr/bin/env python3
"""
轨迹录制脚本
录制三条轨迹:
1. LIO-SAM融合轨迹 (/lio_sam/mapping/odometry)
2. GPS原始轨迹 (/odometry/gps)
3. 纯LIO轨迹 (/lio_sam/mapping/odometry_incremental)
"""

import rospy
import rosbag
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64MultiArray
from lio_sam.msg import MappingStatus
import sys
import signal
import threading
import queue

class TrajectoryRecorder:
    def __init__(self, output_bag):
        self.bag = rosbag.Bag(output_bag, 'w')
        self.fusion_count = 0
        self.gps_count = 0
        self.lio_count = 0
        self.imu_fused_count = 0
        self.imu_incre_count = 0
        self.status_count = 0
        self.degraded_count = 0
        self.bag_ready = True
        self._closed = False

        # NOTE: rospy invokes subscriber callbacks from different threads.
        # rosbag.Bag is NOT thread-safe for concurrent writes; use a single writer thread.
        self._queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        # Keep this non-daemon so a SIGINT shutdown doesn't terminate the process before the
        # writer thread has flushed all queued messages (prevents unindexed/corrupted bags).
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=False)
        self._writer_thread.start()

        rospy.loginfo("Trajectory Recorder started, output: %s", output_bag)

        # Subscribe immediately: under /use_sim_time, rospy.Timer may not fire
        # until /clock starts, which can lead to empty bags if the process exits early.
        self.start_recording(None)

    def start_recording(self, event):
        rospy.Subscriber('/lio_sam/mapping/odometry', Odometry, self.fusion_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental', Odometry, self.lio_callback)
        rospy.Subscriber('/lio_sam/mapping/odometry_incremental_status', MappingStatus, self.status_callback)
        rospy.Subscriber('/odometry/gps', Odometry, self.gps_callback)
        # Optional: hold-out split GPS topics (for unbiased evaluation)
        rospy.Subscriber('/odometry/gps_train', Odometry, self.gps_train_callback)
        rospy.Subscriber('/odometry/gps_test', Odometry, self.gps_test_callback)
        # IMU-preintegration odometry (helps pinpoint whether Z drift comes from IMU or scan-to-map)
        rospy.Subscriber('/odometry/imu', Odometry, self.imu_fused_callback)
        rospy.Subscriber('/odometry/imu_incremental', Odometry, self.imu_incremental_callback)
        rospy.Subscriber('/gnss_degraded', Bool, self.degraded_callback)
        # Detailed debug about GPS factor insertion/weights/residuals (Float64MultiArray)
        rospy.Subscriber('/lio_sam/mapping/gps_factor_debug', Float64MultiArray, self.gps_factor_debug_callback)
        rospy.loginfo("Recording started")

    def _enqueue(self, topic, msg):
        if (not self.bag_ready) or self._stop_event.is_set() or self._closed:
            return False
        try:
            if hasattr(msg, "header"):
                stamp = msg.header.stamp
            else:
                stamp = rospy.Time.now()
            self._queue.put_nowait((topic, msg, stamp))
            return True
        except queue.Full:
            rospy.logwarn_throttle(5.0, "Trajectory Recorder queue full, dropping messages")
            return False

    def fusion_callback(self, msg):
        if self._enqueue('/lio_sam/mapping/odometry', msg):
            self.fusion_count += 1
        if self.fusion_count > 0 and self.fusion_count % 100 == 0:
            rospy.loginfo("Recorded %d fusion poses", self.fusion_count)

    def lio_callback(self, msg):
        if self._enqueue('/lio_sam/mapping/odometry_incremental', msg):
            self.lio_count += 1

    def status_callback(self, msg):
        if self._enqueue('/lio_sam/mapping/odometry_incremental_status', msg):
            self.status_count += 1

    def gps_callback(self, msg):
        if self._enqueue('/odometry/gps', msg):
            self.gps_count += 1

    def gps_train_callback(self, msg):
        # Keep separate topic name in bag so evaluation can compare against the held-out topic.
        if self._enqueue('/odometry/gps_train', msg):
            self.gps_count += 1

    def gps_test_callback(self, msg):
        if self._enqueue('/odometry/gps_test', msg):
            self.gps_count += 1

    def imu_fused_callback(self, msg):
        if self._enqueue('/odometry/imu', msg):
            self.imu_fused_count += 1

    def imu_incremental_callback(self, msg):
        if self._enqueue('/odometry/imu_incremental', msg):
            self.imu_incre_count += 1

    def degraded_callback(self, msg):
        if self._enqueue('/gnss_degraded', msg):
            self.degraded_count += 1

    def gps_factor_debug_callback(self, msg):
        # This topic is low-rate (per keyframe), safe to record verbatim.
        self._enqueue('/lio_sam/mapping/gps_factor_debug', msg)

    def _writer_loop(self):
        while (not self._stop_event.is_set()) or (not self._queue.empty()):
            try:
                topic, msg, stamp = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if stamp is None or (hasattr(stamp, "is_zero") and stamp.is_zero()):
                    stamp = rospy.Time.now()
                self.bag.write(topic, msg, t=stamp)
            except Exception as e:
                rospy.logwarn_throttle(5.0, "Failed to write to bag (%s): %s", topic, str(e))
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.bag_ready = False
        rospy.loginfo("Closing bag file...")
        rospy.loginfo("Total recorded - Fusion: %d, LIO: %d, Status: %d, GPS: %d, IMU: %d, IMU_incre: %d, Degraded: %d",
                      self.fusion_count, self.lio_count, self.status_count, self.gps_count,
                      self.imu_fused_count, self.imu_incre_count, self.degraded_count)
        self._stop_event.set()
        # Drain queue. The writer thread exits when stop_event is set AND the queue is empty.
        # Use a generous timeout because the IMU topics can be high-rate.
        max_wait_s = float(rospy.get_param("~flush_timeout_s", 180.0))
        deadline = rospy.Time.now().to_sec() + max_wait_s
        while self._writer_thread.is_alive() and rospy.Time.now().to_sec() < deadline:
            try:
                qsize = self._queue.qsize()
            except Exception:
                qsize = -1
            rospy.loginfo_throttle(5.0, "Flushing bag writer... queue_size=%s", str(qsize))
            self._writer_thread.join(timeout=1.0)

        if self._writer_thread.is_alive():
            rospy.logwarn("Writer thread did not finish within %.1fs; bag may be incomplete/unindexed", max_wait_s)
        try:
            self.bag.close()
        except Exception as e:
            rospy.logwarn("Failed to close bag cleanly: %s", str(e))

if __name__ == '__main__':
    rospy.init_node('trajectory_recorder')

    if len(sys.argv) < 2:
        output_bag = '/tmp/trajectory_evaluation.bag'
    else:
        output_bag = sys.argv[1]

    recorder = TrajectoryRecorder(output_bag)
    rospy.on_shutdown(recorder.close)

    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        rospy.loginfo("Received signal %d, shutting down gracefully...", sig)
        recorder.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        recorder.close()
