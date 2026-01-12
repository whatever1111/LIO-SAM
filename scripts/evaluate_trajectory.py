#!/usr/bin/env python3
"""
轨迹评估和可视化脚本
分析LIO-SAM融合轨迹与GPS的对比，特别是GNSS降级区间
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation
import sys
import os
import argparse
import math
import yaml


def _load_gps_extrinsic_rot_from_params(params_file):
    """
    Load lio_sam.gpsExtrinsicRot (row-major 3x3) from params.yaml.
    Returns: (R (3x3 np.array) or None, message str)
    """
    try:
        with open(params_file, "r") as f:
            data = yaml.safe_load(f) or {}

        lio = data.get("lio_sam") or {}
        rot = lio.get("gpsExtrinsicRot", None)
        if rot is None:
            return None, "gpsExtrinsicRot not found in params"
        if not isinstance(rot, (list, tuple)) or len(rot) != 9:
            return None, f"gpsExtrinsicRot should be a list of 9 elements, got {type(rot)} len={len(rot) if hasattr(rot, '__len__') else 'n/a'}"

        R = np.array(rot, dtype=float).reshape((3, 3))
        return R, "ok"
    except Exception as e:
        return None, f"failed to load params: {e}"


def _yaw_deg_from_R(R):
    try:
        return math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    except Exception:
        return float("nan")


def _rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def _diag_from_ros_cov36(cov36):
    """
    ROS nav_msgs/Odometry.pose.covariance (row-major 6x6) diag extractor.
    ROS order: [x, y, z, roll, pitch, yaw]
    Returns: [var_x, var_y, var_z, var_roll, var_pitch, var_yaw] or None.
    """
    try:
        if cov36 is None or len(cov36) != 36:
            return None
        # Diagonal indices of row-major 6x6
        return [
            float(cov36[0]),   # xx
            float(cov36[7]),   # yy
            float(cov36[14]),  # zz
            float(cov36[21]),  # roll
            float(cov36[28]),  # pitch
            float(cov36[35]),  # yaw
        ]
    except Exception:
        return None


def _safe_sqrt_var(var):
    try:
        v = float(var)
        if not np.isfinite(v) or v < 0.0:
            return float("nan")
        return float(np.sqrt(v))
    except Exception:
        return float("nan")


def _nearest_idx(t_ref, t_query):
    """
    Find nearest index in sorted t_ref for each t_query.
    Returns: (idx, abs_dt)
    """
    t_ref = np.asarray(t_ref, dtype=float)
    t_query = np.asarray(t_query, dtype=float)
    if t_ref.size == 0 or t_query.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    idx = np.searchsorted(t_ref, t_query, side="left")
    idx0 = np.clip(idx - 1, 0, t_ref.size - 1)
    idx1 = np.clip(idx, 0, t_ref.size - 1)

    dt0 = np.abs(t_ref[idx0] - t_query)
    dt1 = np.abs(t_ref[idx1] - t_query)
    use1 = dt1 < dt0
    best_idx = np.where(use1, idx1, idx0)
    best_dt = np.where(use1, dt1, dt0)
    return best_idx.astype(int), best_dt.astype(float)


def _estimate_se2(gps_xy, ref_xy):
    """
    Estimate rigid SE(2) transform mapping gps_xy -> ref_xy.
    Returns: (R2, t2, yaw_deg)
    """
    gps_xy = np.asarray(gps_xy, dtype=float)
    ref_xy = np.asarray(ref_xy, dtype=float)
    if gps_xy.shape != ref_xy.shape or gps_xy.shape[1] != 2:
        raise ValueError(f"expected Nx2 arrays, got gps={gps_xy.shape}, ref={ref_xy.shape}")
    if gps_xy.shape[0] < 3:
        raise ValueError("need at least 3 points to estimate SE2")

    gps_mean = gps_xy.mean(axis=0)
    ref_mean = ref_xy.mean(axis=0)
    G = gps_xy - gps_mean
    F = ref_xy - ref_mean
    H = G.T @ F
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    t = ref_mean - (R @ gps_mean)
    yaw = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return R, t, yaw


def _interp_xyz(traj, t_samples):
    """
    Interpolate a [t, x, y, z] trajectory at given timestamps.
    Returns: Nx3 array.
    """
    traj = np.asarray(traj, dtype=float)
    t_samples = np.asarray(t_samples, dtype=float)
    if traj.size == 0 or t_samples.size == 0:
        return np.zeros((0, 3), dtype=float)

    t = traj[:, 0]
    out = np.zeros((t_samples.size, 3), dtype=float)
    for i in range(3):
        out[:, i] = np.interp(t_samples, t, traj[:, i + 1])
    return out


def _norm_xy(vec3):
    v = np.asarray(vec3, dtype=float).reshape((-1, 3))
    return np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])

class TrajectoryEvaluator:
    def __init__(self, bag_file, gnss_status_file, gps_topic="/odometry/gps",
                 gps_extrinsic_rot=None, gps_extrinsic_source=None):
        self.bag_file = bag_file
        self.gnss_status_file = gnss_status_file
        self.gps_topic = gps_topic
        self.gps_extrinsic_rot = gps_extrinsic_rot  # 3x3, ENU -> LiDAR/map frame (as used in mapOptmization.cpp)
        self.gps_extrinsic_source = gps_extrinsic_source

        # Data storage
        self.fusion_data = []  # [(t, x, y, z), ...]
        self.gps_data = []      # evaluation-space gps
        self.gps_data_raw = []  # raw gps from bag (before any rotation)
        self.degraded_intervals = []

        # Covariance/flags time series (bag time t.to_sec())
        # diag order: [x, y, z, roll, pitch, yaw] (variance domain)
        self.fusion_cov_diag = []  # [(t, var_x, var_y, var_z, var_r, var_p, var_yaw), ...]
        self.gps_cov_diag = []     # same as above, from /odometry/gps
        self.lio_degenerate = []   # [(t, 0/1), ...] from /lio_sam/mapping/odometry_incremental_status
        self.gps_factor_debug = [] # raw Float64MultiArray rows from /lio_sam/mapping/gps_factor_debug
        self._gps_debug_df = None

        print(f"Loading data from {bag_file}...")
        self.load_data()
        self.load_gnss_status()

    def load_data(self):
        """Load trajectory data from bag file"""
        # Allow reading unindexed bags (in case of improper shutdown)
        bag = rosbag.Bag(self.bag_file, "r", allow_unindexed=True)

        topics = [
            "/lio_sam/mapping/odometry",
            self.gps_topic,
            "/lio_sam/mapping/odometry_incremental_status",
            "/lio_sam/mapping/gps_factor_debug",
        ]

        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()

            if topic == "/lio_sam/mapping/odometry":
                self.fusion_data.append([
                    ts,
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ])
                diag = _diag_from_ros_cov36(getattr(msg.pose, "covariance", None))
                if diag is not None:
                    self.fusion_cov_diag.append([ts] + diag)

            elif topic == "/odometry/gps":
                self.gps_data_raw.append([
                    ts,
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ])
                diag = _diag_from_ros_cov36(getattr(msg.pose, "covariance", None))
                if diag is not None:
                    self.gps_cov_diag.append([ts] + diag)

            elif topic == self.gps_topic:
                # Allow custom GPS topic (e.g. /odometry/gps_test for hold-out evaluation)
                self.gps_data_raw.append([
                    ts,
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ])
                diag = _diag_from_ros_cov36(getattr(msg.pose, "covariance", None))
                if diag is not None:
                    self.gps_cov_diag.append([ts] + diag)

            elif topic == "/lio_sam/mapping/odometry_incremental_status":
                try:
                    self.lio_degenerate.append([ts, 1 if bool(msg.is_degenerate) else 0])
                except Exception:
                    pass

            elif topic == "/lio_sam/mapping/gps_factor_debug":
                try:
                    row = list(getattr(msg, "data", []))
                    if len(row) >= 10:
                        self.gps_factor_debug.append(row)
                except Exception:
                    pass

        bag.close()

        # Sort (recording bags may not be strictly time-ordered)
        self.fusion_data = np.array(self.fusion_data, dtype=float)
        if self.fusion_data.size > 0:
            self.fusion_data = self.fusion_data[np.argsort(self.fusion_data[:, 0])]

        self.gps_data_raw = np.array(self.gps_data_raw, dtype=float)
        if self.gps_data_raw.size > 0:
            self.gps_data_raw = self.gps_data_raw[np.argsort(self.gps_data_raw[:, 0])]

        self.fusion_cov_diag = np.array(self.fusion_cov_diag, dtype=float)
        if self.fusion_cov_diag.size > 0:
            self.fusion_cov_diag = self.fusion_cov_diag[np.argsort(self.fusion_cov_diag[:, 0])]

        self.gps_cov_diag = np.array(self.gps_cov_diag, dtype=float)
        if self.gps_cov_diag.size > 0:
            self.gps_cov_diag = self.gps_cov_diag[np.argsort(self.gps_cov_diag[:, 0])]

        self.lio_degenerate = np.array(self.lio_degenerate, dtype=float)
        if self.lio_degenerate.size > 0:
            self.lio_degenerate = self.lio_degenerate[np.argsort(self.lio_degenerate[:, 0])]

        self.gps_data = self.gps_data_raw.copy()

        print(f"Loaded {len(self.fusion_data)} fusion poses")
        print(f"Loaded {len(self.gps_data_raw)} GPS poses (topic={self.gps_topic})")
        if len(self.fusion_data) > 0:
            print(f"Fusion time range: {self.fusion_data[0,0]:.3f} -> {self.fusion_data[-1,0]:.3f} (dt={self.fusion_data[-1,0]-self.fusion_data[0,0]:.3f}s)")
        if len(self.gps_data_raw) > 0:
            print(f"GPS time range:   {self.gps_data_raw[0,0]:.3f} -> {self.gps_data_raw[-1,0]:.3f} (dt={self.gps_data_raw[-1,0]-self.gps_data_raw[0,0]:.3f}s)")

        # Apply gpsExtrinsicRot if provided: /odometry/gps is in ENU axes, while LIO-SAM map/odom
        # is in LiDAR axes. mapOptmization.cpp applies gpsExtRot to GPS positions before fusing;
        # for evaluation we should compare in the same axis convention.
        if self.gps_extrinsic_rot is not None and self.gps_data.size > 0:
            R = np.asarray(self.gps_extrinsic_rot, dtype=float).reshape((3, 3))
            gps_pos = self.gps_data[:, 1:4]
            gps_pos_rot = gps_pos @ R.T
            self.gps_data[:, 1:4] = gps_pos_rot
            yaw_deg = _yaw_deg_from_R(R)
            src = f" ({self.gps_extrinsic_source})" if self.gps_extrinsic_source else ""
            print(f"Applied gpsExtrinsicRot to GPS positions: yaw={yaw_deg:+.2f} deg{src}")

        self._print_alignment_hint()
        self._print_covariance_hint()

    def _get_gps_debug_df(self):
        if self._gps_debug_df is not None:
            return self._gps_debug_df
        if not self.gps_factor_debug:
            self._gps_debug_df = None
            return None

        cols = [
            "t","kf_idx","gnss_flag_available","gnss_degraded","scan2map_degenerate","loop_factors_added",
            "gps_pos_factor_added","gps_ori_factor_added","skip_reason","gps_noise_scale_this","gps_add_interval_this",
            "pose_cov_x","pose_cov_y","popped_old","rej_nonfinite_cov","rej_cov_threshold","rej_zero","rej_add_interval",
            "gps_var_x_raw","gps_var_y_raw","gps_var_z_raw","gps_var_yaw_raw",
            "gps_var_x_lidar","gps_var_y_lidar","gps_var_z_lidar","gps_var_yaw_lidar",
            "gps_x","gps_y","gps_z","gps_yaw_deg",
            "pre_x","pre_y","pre_z","pre_yaw_deg",
            "post_x","post_y","post_z","post_yaw_deg",
            "delta_x","delta_y","delta_z","delta_yaw_deg",
            "res_pre_norm","res_post_norm","yaw_res_pre_deg","yaw_res_post_deg",
        ]
        rows = []
        for r in self.gps_factor_debug:
            if len(r) < len(cols):
                continue
            rows.append(r[:len(cols)])
        if not rows:
            self._gps_debug_df = None
            return None

        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values("t").reset_index(drop=True)
        self._gps_debug_df = df
        return df

    def _print_covariance_hint(self):
        def _series_summary(name, arr):
            if arr is None or not hasattr(arr, "shape") or arr.size == 0:
                print(f"[cov] {name}: no data")
                return

            # arr columns: t, x, y, z, roll, pitch, yaw (variances)
            pos = arr[:, 1:4]
            valid = np.all(np.isfinite(pos), axis=1) & np.all(pos > 0.0, axis=1)
            ratio = float(np.mean(valid)) if valid.size else 0.0
            if np.any(valid):
                mean_std = np.sqrt(np.mean(pos[valid], axis=0))
                print(f"[cov] {name}: {arr.shape[0]} msgs, valid_pos_diag={ratio*100:.1f}%, mean_std_xyz=[{mean_std[0]:.3g},{mean_std[1]:.3g},{mean_std[2]:.3g}] m")
            else:
                print(f"[cov] {name}: {arr.shape[0]} msgs, valid_pos_diag={ratio*100:.1f}%")

        _series_summary(f"GPS({self.gps_topic})", self.gps_cov_diag)
        _series_summary("Fusion(/lio_sam/mapping/odometry)", self.fusion_cov_diag)

    def _print_alignment_hint(self):
        if len(self.fusion_data) == 0 or len(self.gps_data_raw) == 0:
            return

        try:
            # Compare fusion vs raw gps (to reveal axis mismatch), and fusion vs eval gps (after rotation)
            t_common, fusion_interp, gps_raw_interp = self.interpolate_trajectory(self.fusion_data, self.gps_data_raw)
            if len(t_common) < 10:
                return

            err_raw = np.linalg.norm(fusion_interp - gps_raw_interp, axis=1)
            R_raw, t_raw, yaw_raw = _estimate_se2(gps_raw_interp[:, :2], fusion_interp[:, :2])
            print(f"[alignment] raw GPS->fusion best-fit yaw={yaw_raw:+.2f} deg, trans=[{t_raw[0]:+.3f},{t_raw[1]:+.3f}] m, rmse={_rmse(err_raw):.3f} m")

            t_common2, fusion_interp2, gps_eval_interp = self.interpolate_trajectory(self.fusion_data, self.gps_data)
            if len(t_common2) < 10:
                return
            err_eval = np.linalg.norm(fusion_interp2 - gps_eval_interp, axis=1)
            R_eval, t_eval, yaw_eval = _estimate_se2(gps_eval_interp[:, :2], fusion_interp2[:, :2])
            print(f"[alignment] eval GPS->fusion best-fit yaw={yaw_eval:+.2f} deg, trans=[{t_eval[0]:+.3f},{t_eval[1]:+.3f}] m, rmse={_rmse(err_eval):.3f} m")
        except Exception as e:
            print(f"Warning: alignment hint failed: {e}")

    def load_gnss_status(self):
        """Load GNSS degradation status"""
        # Prefer using /gnss_degraded recorded in the evaluation bag, because it shares the same time base
        # as other recorded topics (and avoids header.stamp vs /clock mismatch).
        try:
            bag = rosbag.Bag(self.bag_file, 'r', allow_unindexed=True)
            timestamps = []
            degraded = []
            for _, msg, t in bag.read_messages(topics=['/gnss_degraded']):
                timestamps.append(t.to_sec())
                degraded.append(bool(msg.data))
            bag.close()

            if len(timestamps) > 0:
                degraded = np.array(degraded, dtype=bool)
                timestamps = np.array(timestamps, dtype=float)
                print(f"Loaded {len(timestamps)} /gnss_degraded messages from bag")
                print(f"/gnss_degraded time range: {timestamps[0]:.3f} -> {timestamps[-1]:.3f} (dt={timestamps[-1]-timestamps[0]:.3f}s)")
            else:
                timestamps = None
                degraded = None
        except Exception as e:
            print(f"Warning: failed to read /gnss_degraded from bag: {e}")
            timestamps = None
            degraded = None

        if timestamps is None or degraded is None:
            if not os.path.exists(self.gnss_status_file):
                print(f"Warning: GNSS status file not found: {self.gnss_status_file}")
                return

            df = pd.read_csv(self.gnss_status_file)

            # Find degradation intervals
            degraded = df['is_degraded'].values.astype(bool)
            # Backward-compatible: old logs used "timestamp", new logs use "ros_time"
            if 'ros_time' in df.columns:
                timestamps = df['ros_time'].values
            else:
                timestamps = df['timestamp'].values

        in_degradation = False
        start_time = None

        for i, (is_deg, t) in enumerate(zip(degraded, timestamps)):
            if is_deg and not in_degradation:
                # Start of degradation
                start_time = t
                in_degradation = True
            elif not is_deg and in_degradation:
                # End of degradation
                self.degraded_intervals.append((start_time, t))
                in_degradation = False

        # If still degraded at end
        if in_degradation:
            self.degraded_intervals.append((start_time, timestamps[-1]))

        print(f"Found {len(self.degraded_intervals)} GNSS degradation intervals:")
        for i, (start, end) in enumerate(self.degraded_intervals):
            print(f"  Interval {i+1}: {start:.3f} - {end:.3f} ({end-start:.3f}s)")

    def interpolate_trajectory(self, traj_query, traj_ref):
        """Interpolate trajectory to match timestamps"""
        t_query = traj_query[:, 0]
        t_ref = traj_ref[:, 0]

        # Find common time range
        t_start = max(t_query[0], t_ref[0])
        t_end = min(t_query[-1], t_ref[-1])

        # Interpolate on reference timestamps
        indices = (t_ref >= t_start) & (t_ref <= t_end)
        t_common = t_ref[indices]

        query_interp = np.zeros((len(t_common), 3))
        for i in range(3):
            query_interp[:, i] = np.interp(t_common, t_query, traj_query[:, i+1])

        ref_interp = traj_ref[indices, 1:4]

        return t_common, query_interp, ref_interp

    def compute_errors(self):
        """Compute position errors between fusion and GPS"""
        t_common, fusion_interp, gps_interp = self.interpolate_trajectory(
            self.fusion_data, self.gps_data
        )

        # Compute errors
        errors = np.linalg.norm(fusion_interp - gps_interp, axis=1)

        return t_common, errors, fusion_interp, gps_interp

    def compute_degraded_bridge_drift(self, max_anchor_gap_s=0.5):
        """
        Evaluate degraded-interval drift using GNSS-good anchors outside the degraded window.

        For each degraded interval [start, end]:
          - choose t_pre = last GPS sample with t <= start
          - choose t_post = first GPS sample with t >= end
          - compare relative displacement between (t_pre, t_post): (fusion_delta - gps_delta)

        This is the recommended "leave-one-out" style metric when you trust GNSS-good but distrust
        GNSS-degraded segments, because it does NOT assume degraded GNSS is correct.
        """
        if not self.degraded_intervals:
            return None
        if self.gps_data is None or self.gps_data.size == 0:
            return None
        if self.fusion_data is None or self.fusion_data.size == 0:
            return None

        t_gps = self.gps_data[:, 0]
        t_fus = self.fusion_data[:, 0]
        if t_gps.size < 2 or t_fus.size < 2:
            return None

        dbg = self._get_gps_debug_df()

        rows = []
        for i, (start, end) in enumerate(self.degraded_intervals):
            if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                continue

            idx_pre = int(np.searchsorted(t_gps, start, side="right") - 1)
            idx_post = int(np.searchsorted(t_gps, end, side="left"))
            if idx_pre < 0 or idx_post >= t_gps.size:
                continue

            t_pre = float(t_gps[idx_pre])
            t_post = float(t_gps[idx_post])
            gap_pre = float(start - t_pre)
            gap_post = float(t_post - end)

            # Interpolate fusion at anchor timestamps
            p_f = _interp_xyz(self.fusion_data, np.array([t_pre, t_post], dtype=float))
            if p_f.shape[0] != 2:
                continue
            p_f_pre = p_f[0]
            p_f_post = p_f[1]

            p_g_pre = self.gps_data[idx_pre, 1:4].astype(float)
            p_g_post = self.gps_data[idx_post, 1:4].astype(float)

            d_f = p_f_post - p_f_pre
            d_g = p_g_post - p_g_pre
            d_err = d_f - d_g

            dur = float(end - start)
            gps_dist_xy = float(np.hypot(d_g[0], d_g[1]))
            fus_dist_xy = float(np.hypot(d_f[0], d_f[1]))
            drift_xy = float(np.hypot(d_err[0], d_err[1]))
            drift_3d = float(np.linalg.norm(d_err))

            anchor_err_pre = float(np.linalg.norm(p_f_pre - p_g_pre))
            anchor_err_post = float(np.linalg.norm(p_f_post - p_g_post))

            gps_added = None
            dominant_skip = None
            if dbg is not None:
                in_int = (dbg["t"] >= start) & (dbg["t"] <= end)
                try:
                    gps_added = int(dbg.loc[in_int, "gps_pos_factor_added"].sum())
                except Exception:
                    gps_added = None
                try:
                    if int(in_int.sum()) > 0:
                        dominant_skip = int(dbg.loc[in_int, "skip_reason"].mode().iloc[0])
                except Exception:
                    dominant_skip = None

            rows.append({
                "region": i + 1,
                "deg_start": start,
                "deg_end": end,
                "duration_s": dur,
                "t_pre": t_pre,
                "t_post": t_post,
                "gap_pre_s": gap_pre,
                "gap_post_s": gap_post,
                "anchor_gap_ok": 1 if (gap_pre <= max_anchor_gap_s and gap_post <= max_anchor_gap_s) else 0,
                "gps_dx": float(d_g[0]),
                "gps_dy": float(d_g[1]),
                "gps_dz": float(d_g[2]),
                "gps_dist_xy": gps_dist_xy,
                "fusion_dx": float(d_f[0]),
                "fusion_dy": float(d_f[1]),
                "fusion_dz": float(d_f[2]),
                "fusion_dist_xy": fus_dist_xy,
                "drift_dx": float(d_err[0]),
                "drift_dy": float(d_err[1]),
                "drift_dz": float(d_err[2]),
                "drift_xy": drift_xy,
                "drift_3d": drift_3d,
                "drift_rate_mps": float(drift_xy / dur) if dur > 1e-6 else float("nan"),
                "drift_per_meter": float(drift_xy / gps_dist_xy) if gps_dist_xy > 1e-3 else float("nan"),
                "anchor_err_pre": anchor_err_pre,
                "anchor_err_post": anchor_err_post,
                "gps_factors_added_in_degraded": gps_added,
                "dominant_skip_reason_in_degraded": dominant_skip,
            })

        if not rows:
            return None
        return pd.DataFrame(rows)

    def plot_trajectories_3d(self, output_file='trajectory_3d.png'):
        """Plot 3D trajectories"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot full trajectories
        ax.plot(self.gps_data[:, 1], self.gps_data[:, 2], self.gps_data[:, 3],
                'g-', label='GPS (eval)', linewidth=2, alpha=0.7)
        ax.plot(self.fusion_data[:, 1], self.fusion_data[:, 2], self.fusion_data[:, 3],
                'b-', label='Fusion (LIO-SAM)', linewidth=2, alpha=0.7)

        # Highlight degraded intervals
        for start, end in self.degraded_intervals:
            # Find indices
            idx_fusion = (self.fusion_data[:, 0] >= start) & (self.fusion_data[:, 0] <= end)
            if np.any(idx_fusion):
                ax.plot(self.fusion_data[idx_fusion, 1],
                       self.fusion_data[idx_fusion, 2],
                       self.fusion_data[idx_fusion, 3],
                       'r-', linewidth=3, alpha=0.8)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        title = '3D Trajectory Comparison (Red = GNSS Degraded)'
        if self.gps_extrinsic_rot is not None:
            title += ' [GPS rotated by gpsExtrinsicRot]'
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved 3D trajectory plot to {output_file}")
        plt.close()

    def plot_trajectories_2d(self, output_file='trajectory_2d.png'):
        """Plot 2D top-down view"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot full trajectories
        ax.plot(self.gps_data[:, 1], self.gps_data[:, 2],
                'g-', label='GPS (eval)', linewidth=2, alpha=0.7)
        ax.plot(self.fusion_data[:, 1], self.fusion_data[:, 2],
                'b-', label='Fusion (LIO-SAM)', linewidth=2, alpha=0.7)

        # Highlight degraded intervals
        for i, (start, end) in enumerate(self.degraded_intervals):
            idx_fusion = (self.fusion_data[:, 0] >= start) & (self.fusion_data[:, 0] <= end)
            if np.any(idx_fusion):
                label = 'GNSS Degraded' if i == 0 else None
                ax.plot(self.fusion_data[idx_fusion, 1],
                       self.fusion_data[idx_fusion, 2],
                       'r-', linewidth=3, alpha=0.8, label=label)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        title = '2D Trajectory Comparison (Top View)'
        if self.gps_extrinsic_rot is not None:
            title += ' [GPS rotated by gpsExtrinsicRot]'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved 2D trajectory plot to {output_file}")
        plt.close()

    def plot_errors(self, output_file='errors.png'):
        """Plot position errors over time"""
        t_common, errors, _, _ = self.compute_errors()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot errors
        t0 = float(t_common[0]) if len(t_common) else 0.0
        t1 = float(t_common[-1]) if len(t_common) else 0.0
        t_rel = t_common - t0
        ax1.plot(t_rel, errors, 'b-', linewidth=1.5, alpha=0.7)

        # Highlight degraded intervals
        # Clip to the plotted time range and avoid negative spans. Also, if the first degraded
        # sample arrives slightly after t0 (state topic delay), start the shaded region at 0.
        eps_start = 0.5  # seconds
        shaded_once = False
        for start, end in self.degraded_intervals:
            s = float(start)
            e = float(end)
            if not np.isfinite(s) or not np.isfinite(e) or e <= s:
                continue

            # Clip to [t0, t1]
            s_clip = max(s, t0)
            e_clip = min(e, t1)
            if e_clip <= s_clip:
                continue

            s_rel = s_clip - t0
            e_rel = e_clip - t0

            # If the interval starts just after t0, assume degraded at t0 as well.
            if 0.0 < s_rel < eps_start:
                s_rel = 0.0

            ax1.axvspan(
                s_rel,
                e_rel,
                color="red",
                alpha=0.2,
                label="GNSS Degraded" if not shaded_once else None,
            )
            shaded_once = True

        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Position Error (m)', fontsize=12)
        ax1.set_title('Position Error: Fusion vs GPS', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Export error series CSV next to the plot for deeper offline debugging.
        try:
            out_csv = os.path.splitext(output_file)[0] + ".csv"
            degraded_flags = np.zeros(len(t_common), dtype=int)
            for i, (start, end) in enumerate(self.degraded_intervals):
                s = float(start)
                e = float(end)
                if not np.isfinite(s) or not np.isfinite(e) or e <= s:
                    continue
                degraded_flags |= ((t_common >= s) & (t_common <= e)).astype(int)

            df_err = pd.DataFrame({
                "t": t_common.astype(float),
                "t_rel": t_rel.astype(float),
                "error_m": np.asarray(errors, dtype=float),
                "gnss_degraded": degraded_flags.astype(int),
            })
            df_err.to_csv(out_csv, index=False)
            print(f"Saved error CSV to {out_csv}")
        except Exception as e:
            print(f"Warning: failed to write error CSV: {e}")

        # Plot cumulative error distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax2.plot(sorted_errors, cumulative, 'b-', linewidth=2)
        ax2.set_xlabel('Position Error (m)', fontsize=12)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.set_title('Cumulative Error Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Mark percentiles
        for percentile in [50, 90, 95, 99]:
            value = np.percentile(errors, percentile)
            ax2.axvline(value, color='r', linestyle='--', alpha=0.5)
            ax2.text(value, percentile, f'{percentile}%: {value:.3f}m',
                    rotation=90, va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved error plot to {output_file}")
        plt.close()

    def plot_covariances(self, output_file="covariances.png", output_csv="covariances.csv"):
        """Plot GNSS degraded flag vs covariance curves (GPS + LIO-SAM)."""
        if (self.gps_cov_diag is None or self.gps_cov_diag.size == 0) and (
            self.fusion_cov_diag is None or self.fusion_cov_diag.size == 0
        ):
            print("No covariance data found in bag, skipping covariance plots.")
            return

        # Choose a common t0 for plotting
        t0_candidates = []
        if self.fusion_data is not None and self.fusion_data.size > 0:
            t0_candidates.append(float(self.fusion_data[0, 0]))
        if self.gps_data is not None and self.gps_data.size > 0:
            t0_candidates.append(float(self.gps_data[0, 0]))
        if self.degraded_intervals:
            t0_candidates.append(float(self.degraded_intervals[0][0]))
        if self.lio_degenerate is not None and self.lio_degenerate.size > 0:
            t0_candidates.append(float(self.lio_degenerate[0, 0]))
        t0 = min(t0_candidates) if t0_candidates else 0.0

        fig, axs = plt.subplots(3, 1, figsize=(16, 11), sharex=True)

        def shade_degraded(ax):
            for i, (start, end) in enumerate(self.degraded_intervals):
                ax.axvspan(start - t0, end - t0, color="red", alpha=0.15, label="GNSS degraded" if i == 0 else None)

        # 1) Flags
        ax0 = axs[0]
        shade_degraded(ax0)
        if self.lio_degenerate is not None and self.lio_degenerate.size > 0:
            ax0.step(self.lio_degenerate[:, 0] - t0, self.lio_degenerate[:, 1], where="post", color="k", linewidth=1.2, label="LIO scan2map degenerate")
        ax0.set_ylim(-0.1, 1.1)
        ax0.set_ylabel("Flag")
        ax0.set_title("Flags vs Time")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right")

        # 2) Position std-dev (sqrt(variance))
        ax1 = axs[1]
        shade_degraded(ax1)
        if self.gps_cov_diag is not None and self.gps_cov_diag.size > 0:
            t = self.gps_cov_diag[:, 0] - t0
            sx = np.array([_safe_sqrt_var(v) for v in self.gps_cov_diag[:, 1]])
            sy = np.array([_safe_sqrt_var(v) for v in self.gps_cov_diag[:, 2]])
            sz = np.array([_safe_sqrt_var(v) for v in self.gps_cov_diag[:, 3]])
            ax1.plot(t, sx, color="g", linewidth=1.0, alpha=0.8, label="GPS σx (m)")
            ax1.plot(t, sy, color="g", linewidth=1.0, alpha=0.5, label="GPS σy (m)")
            ax1.plot(t, sz, color="g", linewidth=1.0, alpha=0.3, label="GPS σz (m)")
        if self.fusion_cov_diag is not None and self.fusion_cov_diag.size > 0:
            t = self.fusion_cov_diag[:, 0] - t0
            sx = np.array([_safe_sqrt_var(v) for v in self.fusion_cov_diag[:, 1]])
            sy = np.array([_safe_sqrt_var(v) for v in self.fusion_cov_diag[:, 2]])
            sz = np.array([_safe_sqrt_var(v) for v in self.fusion_cov_diag[:, 3]])
            ax1.plot(t, sx, color="b", linewidth=1.0, alpha=0.8, label="Fusion σx (m)")
            ax1.plot(t, sy, color="b", linewidth=1.0, alpha=0.5, label="Fusion σy (m)")
            ax1.plot(t, sz, color="b", linewidth=1.0, alpha=0.3, label="Fusion σz (m)")
        ax1.set_ylabel("σ (m)")
        ax1.set_title("Position Uncertainty (std-dev from covariance diag)")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")
        ax1.legend(ncol=2, fontsize=9, loc="upper right")

        # 3) Yaw std-dev (deg)
        ax2 = axs[2]
        shade_degraded(ax2)
        rad2deg = 180.0 / math.pi
        if self.gps_cov_diag is not None and self.gps_cov_diag.size > 0:
            t = self.gps_cov_diag[:, 0] - t0
            syaw = np.array([_safe_sqrt_var(v) * rad2deg for v in self.gps_cov_diag[:, 6]])
            ax2.plot(t, syaw, color="g", linewidth=1.2, alpha=0.8, label="GPS σyaw (deg)")
        if self.fusion_cov_diag is not None and self.fusion_cov_diag.size > 0:
            t = self.fusion_cov_diag[:, 0] - t0
            syaw = np.array([_safe_sqrt_var(v) * rad2deg for v in self.fusion_cov_diag[:, 6]])
            ax2.plot(t, syaw, color="b", linewidth=1.2, alpha=0.8, label="Fusion σyaw (deg)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("σyaw (deg)")
        ax2.set_title("Yaw Uncertainty (std-dev)")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved covariance plot to {output_file}")
        plt.close()

        # Also export a time-aligned CSV (base = fusion timestamps if present; else GPS).
        base_times = None
        base_src = None
        if self.fusion_cov_diag is not None and self.fusion_cov_diag.size > 0:
            base_times = self.fusion_cov_diag[:, 0]
            base_src = "fusion"
        elif self.gps_cov_diag is not None and self.gps_cov_diag.size > 0:
            base_times = self.gps_cov_diag[:, 0]
            base_src = "gps"
        else:
            return

        # Determine degraded flag for each base time via intervals.
        degraded_flags = np.zeros(base_times.shape[0], dtype=int)
        if self.degraded_intervals:
            for i, t in enumerate(base_times):
                for start, end in self.degraded_intervals:
                    if start <= t <= end:
                        degraded_flags[i] = 1
                        break

        match_max_dt = 0.2  # seconds

        def match_series(series):
            if series is None or series.size == 0:
                return None, None
            idx, dt = _nearest_idx(series[:, 0], base_times)
            matched = series[idx]
            valid = dt <= match_max_dt
            return matched, valid

        gps_match, gps_valid = match_series(self.gps_cov_diag)
        fusion_match, fusion_valid = match_series(self.fusion_cov_diag)
        deg_match, deg_valid = match_series(self.lio_degenerate)

        df = pd.DataFrame({
            "t": base_times,
            "t_rel": base_times - t0,
            "base": base_src,
            "gnss_degraded": degraded_flags.astype(int),
        })

        def add_cov_cols(prefix, match, valid):
            if match is None:
                for k in ["var_x","var_y","var_z","var_roll","var_pitch","var_yaw"]:
                    df[f"{prefix}_{k}"] = np.nan
                df[f"{prefix}_valid"] = 0
                return

            df[f"{prefix}_var_x"] = match[:, 1]
            df[f"{prefix}_var_y"] = match[:, 2]
            df[f"{prefix}_var_z"] = match[:, 3]
            df[f"{prefix}_var_roll"] = match[:, 4]
            df[f"{prefix}_var_pitch"] = match[:, 5]
            df[f"{prefix}_var_yaw"] = match[:, 6]
            df[f"{prefix}_valid"] = valid.astype(int)

        add_cov_cols("gps", gps_match, gps_valid if gps_valid is not None else np.zeros(df.shape[0], dtype=bool))
        add_cov_cols("fusion", fusion_match, fusion_valid if fusion_valid is not None else np.zeros(df.shape[0], dtype=bool))

        if deg_match is not None:
            df["lio_degenerate"] = np.where(deg_valid, deg_match[:, 1].astype(int), 0)
            df["lio_degenerate_valid"] = deg_valid.astype(int)
        else:
            df["lio_degenerate"] = 0
            df["lio_degenerate_valid"] = 0

        df.to_csv(output_csv, index=False)
        print(f"Saved covariance CSV to {output_csv}")

    def plot_gps_factor_debug(self, output_file="gps_factor_debug.png", output_csv="gps_factor_debug.csv"):
        """Plot GPS factor decision/weight/residual debug emitted by mapOptmization."""
        if not self.gps_factor_debug:
            print("No /lio_sam/mapping/gps_factor_debug found in bag, skipping GPS debug plots.")
            return

        # Expected schema (Float64MultiArray length >= 46)
        cols = [
            "t","kf_idx","gnss_flag_available","gnss_degraded","scan2map_degenerate","loop_factors_added",
            "gps_pos_factor_added","gps_ori_factor_added","skip_reason","gps_noise_scale_this","gps_add_interval_this",
            "pose_cov_x","pose_cov_y","popped_old","rej_nonfinite_cov","rej_cov_threshold","rej_zero","rej_add_interval",
            "gps_var_x_raw","gps_var_y_raw","gps_var_z_raw","gps_var_yaw_raw",
            "gps_var_x_lidar","gps_var_y_lidar","gps_var_z_lidar","gps_var_yaw_lidar",
            "gps_x","gps_y","gps_z","gps_yaw_deg",
            "pre_x","pre_y","pre_z","pre_yaw_deg",
            "post_x","post_y","post_z","post_yaw_deg",
            "delta_x","delta_y","delta_z","delta_yaw_deg",
            "res_pre_norm","res_post_norm","yaw_res_pre_deg","yaw_res_post_deg",
        ]
        rows = []
        for r in self.gps_factor_debug:
            if len(r) < len(cols):
                continue
            rows.append(r[:len(cols)])

        if not rows:
            print("GPS debug topic present but no valid rows matched expected schema, skipping.")
            return

        df = pd.DataFrame(rows, columns=cols)
        df = df.sort_values("t").reset_index(drop=True)

        # Relative time base
        t0_candidates = []
        if self.fusion_data is not None and self.fusion_data.size > 0:
            t0_candidates.append(float(self.fusion_data[0, 0]))
        if df.shape[0] > 0:
            t0_candidates.append(float(df["t"].iloc[0]))
        if self.degraded_intervals:
            t0_candidates.append(float(self.degraded_intervals[0][0]))
        t0 = min(t0_candidates) if t0_candidates else float(df["t"].iloc[0])
        df["t_rel"] = df["t"] - t0

        # Save CSV for deeper offline analysis
        df.to_csv(output_csv, index=False)
        print(f"Saved GPS factor debug CSV to {output_csv}")

        # Plot
        fig, axs = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

        def shade_degraded(ax):
            for i, (start, end) in enumerate(self.degraded_intervals):
                ax.axvspan(start - t0, end - t0, color="red", alpha=0.12, label="GNSS degraded" if i == 0 else None)

        # 1) Decisions/flags
        ax0 = axs[0]
        shade_degraded(ax0)
        ax0.scatter(df["t_rel"], df["skip_reason"], s=10, c="gray", alpha=0.7, label="skip_reason(code)")
        added = df["gps_pos_factor_added"] > 0.5
        ax0.scatter(df.loc[added, "t_rel"], df.loc[added, "skip_reason"] * 0.0, s=14, c="g", alpha=0.9, label="GPS pos factor added")
        deg = df["scan2map_degenerate"] > 0.5
        ax0.scatter(df.loc[deg, "t_rel"], df.loc[deg, "skip_reason"] * 0.0 + 1.0, s=14, c="k", alpha=0.6, label="scan2map degenerate")
        loop = df["loop_factors_added"] > 0.5
        ax0.scatter(df.loc[loop, "t_rel"], df.loc[loop, "skip_reason"] * 0.0 + 2.0, s=14, c="m", alpha=0.6, label="loop factors added")
        ax0.set_ylabel("Code/Flag")
        ax0.set_title("GPS Factor Decision Trace (codes)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right", fontsize=9, ncol=2)

        # 2) Weighting knobs over time
        ax1 = axs[1]
        shade_degraded(ax1)
        ax1.plot(df["t_rel"], df["gps_noise_scale_this"], color="b", linewidth=1.2, alpha=0.9, label="gpsNoiseScaleThis")
        ax1.set_yscale("log")
        ax1.set_ylabel("Scale (log)")
        ax1.set_title("GPS Weight Control Actually Used (gpsNoiseScaleThis)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # 3) Residuals: GPS - pose (pre/post update)
        ax2 = axs[2]
        shade_degraded(ax2)
        ax2.plot(df["t_rel"], df["res_pre_norm"], color="orange", linewidth=1.0, alpha=0.8, label="|GPS - pre_pose| (m)")
        ax2.plot(df["t_rel"], df["res_post_norm"], color="green", linewidth=1.2, alpha=0.8, label="|GPS - post_pose| (m)")
        ax2.set_ylabel("m")
        ax2.set_yscale("log")
        ax2.set_title("GPS Residual Before/After ISAM2 Update")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        # 4) Correction magnitude (post - pre)
        ax3 = axs[3]
        shade_degraded(ax3)
        dpos = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2 + df["delta_z"]**2)
        ax3.plot(df["t_rel"], dpos, color="purple", linewidth=1.2, alpha=0.9, label="|post-pre| position (m)")
        ax3.plot(df["t_rel"], np.abs(df["delta_yaw_deg"]), color="brown", linewidth=1.0, alpha=0.7, label="|post-pre| yaw (deg)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("m / deg")
        ax3.set_yscale("log")
        ax3.set_title("How Much Optimization Pulled the Pose at Each Keyframe")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved GPS factor debug plot to {output_file}")
        plt.close()

    def plot_degraded_regions(self, output_dir='degraded_regions'):
        """Plot detailed views of degraded regions"""
        os.makedirs(output_dir, exist_ok=True)

        t_common, errors, fusion_interp, gps_interp = self.compute_errors()

        for i, (start, end) in enumerate(self.degraded_intervals):
            # Add buffer before and after
            buffer = 10.0  # seconds
            start_buffered = start - buffer
            end_buffered = end + buffer

            # Find indices
            idx = (t_common >= start_buffered) & (t_common <= end_buffered)
            idx_degraded = (t_common >= start) & (t_common <= end)

            if not np.any(idx):
                continue

            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 2D trajectory
            ax1.plot(gps_interp[idx, 0], gps_interp[idx, 1],
                    'g-', label='GPS', linewidth=2, alpha=0.7)
            ax1.plot(fusion_interp[idx, 0], fusion_interp[idx, 1],
                    'b-', label='Fusion', linewidth=2, alpha=0.7)
            ax1.plot(fusion_interp[idx_degraded, 0], fusion_interp[idx_degraded, 1],
                    'r-', label='Degraded Period', linewidth=3, alpha=0.8)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title(f'Degraded Region {i+1}: Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Error over time
            t_plot = t_common[idx] - t_common[idx][0]
            ax2.plot(t_plot, errors[idx], 'b-', linewidth=2)
            degraded_mask = idx_degraded & idx
            ax2.axvspan(start - t_common[idx][0], end - t_common[idx][0],
                       color='red', alpha=0.2, label='GNSS Degraded')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position Error (m)')
            ax2.set_title(f'Degraded Region {i+1}: Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'degraded_region_{i+1}.png')
            plt.savefig(output_file, dpi=300)
            print(f"Saved degraded region {i+1} plot to {output_file}")
            plt.close()

    def compute_statistics(self, output_file='statistics.txt'):
        """Compute and save error statistics"""
        t_common, errors, _, _ = self.compute_errors()

        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LIO-SAM GNSS Fusion Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            # Global statistics
            f.write("Global Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total trajectory points: {len(errors)}\n")
            f.write(f"Mean error: {np.mean(errors):.4f} m\n")
            f.write(f"Std dev: {np.std(errors):.4f} m\n")
            f.write(f"RMS error: {np.sqrt(np.mean(errors**2)):.4f} m\n")
            f.write(f"Max error: {np.max(errors):.4f} m\n")
            f.write(f"Median error: {np.median(errors):.4f} m\n")
            f.write(f"50th percentile: {np.percentile(errors, 50):.4f} m\n")
            f.write(f"90th percentile: {np.percentile(errors, 90):.4f} m\n")
            f.write(f"95th percentile: {np.percentile(errors, 95):.4f} m\n")
            f.write(f"99th percentile: {np.percentile(errors, 99):.4f} m\n\n")

            # Degraded region statistics
            f.write("GNSS Degraded Region Statistics:\n")
            f.write("-" * 60 + "\n")

            for i, (start, end) in enumerate(self.degraded_intervals):
                idx = (t_common >= start) & (t_common <= end)
                if not np.any(idx):
                    continue

                errors_deg = errors[idx]
                duration = end - start

                f.write(f"\nRegion {i+1}:\n")
                f.write(f"  Duration: {duration:.2f} s\n")
                f.write(f"  Points: {len(errors_deg)}\n")
                f.write(f"  Mean error: {np.mean(errors_deg):.4f} m\n")
                f.write(f"  RMS error: {np.sqrt(np.mean(errors_deg**2)):.4f} m\n")
                f.write(f"  Max error: {np.max(errors_deg):.4f} m\n")
                f.write(f"  Median error: {np.median(errors_deg):.4f} m\n")

            # Non-degraded statistics
            f.write("\n" + "=" * 60 + "\n")
            f.write("Non-Degraded Region Statistics:\n")
            f.write("-" * 60 + "\n")

            # Create mask for non-degraded regions
            non_degraded_mask = np.ones(len(t_common), dtype=bool)
            for start, end in self.degraded_intervals:
                idx = (t_common >= start) & (t_common <= end)
                non_degraded_mask &= ~idx

            if np.any(non_degraded_mask):
                errors_nondeg = errors[non_degraded_mask]
                f.write(f"Points: {len(errors_nondeg)}\n")
                f.write(f"Mean error: {np.mean(errors_nondeg):.4f} m\n")
                f.write(f"RMS error: {np.sqrt(np.mean(errors_nondeg**2)):.4f} m\n")
                f.write(f"Max error: {np.max(errors_nondeg):.4f} m\n")
                f.write(f"Median error: {np.median(errors_nondeg):.4f} m\n")

            # Degraded interval drift using GNSS-good anchors (recommended for hold-out evaluation)
            df_bridge = self.compute_degraded_bridge_drift(max_anchor_gap_s=0.5)
            f.write("\n" + "=" * 60 + "\n")
            f.write("Degraded Interval Drift (Anchored by GNSS-Good GPS)\n")
            f.write("-" * 60 + "\n")
            f.write("Metric: compare relative displacement between anchors around the degraded window:\n")
            f.write("  drift = (fusion(t_post)-fusion(t_pre)) - (gps(t_post)-gps(t_pre))\n")
            f.write("This does NOT assume degraded GNSS is correct; it only uses GNSS-good anchors.\n")
            f.write("Note: anchor_err_* includes any constant lever-arm/frame offset.\n\n")

            if df_bridge is None or df_bridge.empty:
                f.write("No valid degraded intervals with usable anchors.\n")
            else:
                # Summaries over intervals where anchors are close to boundaries
                ok = df_bridge["anchor_gap_ok"] > 0.5
                df_ok = df_bridge[ok].copy()
                f.write(f"Intervals total: {len(df_bridge)}\n")
                f.write(f"Intervals with anchor_gap_ok (<=0.5s each side): {len(df_ok)}\n\n")

                def _wstats(name, arr):
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        return f"{name}: n=0\n"
                    return (
                        f"{name}: n={arr.size} mean={np.mean(arr):.4f} "
                        f"median={np.median(arr):.4f} p90={np.percentile(arr,90):.4f} max={np.max(arr):.4f}\n"
                    )

                f.write(_wstats("drift_xy (m)", df_ok["drift_xy"] if not df_ok.empty else df_bridge["drift_xy"]))
                f.write(_wstats("drift_rate (m/s)", df_ok["drift_rate_mps"] if not df_ok.empty else df_bridge["drift_rate_mps"]))
                f.write(_wstats("drift_per_meter (m/m)", df_ok["drift_per_meter"] if not df_ok.empty else df_bridge["drift_per_meter"]))
                f.write("\nPer-interval (bridge) details:\n")
                for _, r in df_bridge.iterrows():
                    f.write(
                        f"Region {int(r['region'])}: dur={r['duration_s']:.2f}s "
                        f"gap_pre={r['gap_pre_s']:.3f}s gap_post={r['gap_post_s']:.3f}s "
                        f"drift_xy={r['drift_xy']:.3f}m drift_rate={r['drift_rate_mps']:.4f}m/s "
                        f"gps_dist_xy={r['gps_dist_xy']:.3f}m "
                        f"gps_factors_in_degraded={r['gps_factors_added_in_degraded']}\n"
                    )

        print(f"Saved statistics to {output_file}")

        # Also print to console
        with open(output_file, 'r') as f:
            print("\n" + f.read())

    def run_evaluation(self, output_dir='.'):
        """Run complete evaluation"""
        print("\n" + "=" * 60)
        print("Starting Trajectory Evaluation")
        print("=" * 60 + "\n")

        os.makedirs(output_dir, exist_ok=True)

        self.plot_trajectories_3d(os.path.join(output_dir, 'trajectory_3d.png'))
        self.plot_trajectories_2d(os.path.join(output_dir, 'trajectory_2d.png'))
        self.plot_errors(os.path.join(output_dir, 'errors.png'))
        self.plot_covariances(os.path.join(output_dir, "covariances.png"),
                              os.path.join(output_dir, "covariances.csv"))
        self.plot_gps_factor_debug(os.path.join(output_dir, "gps_factor_debug.png"),
                                   os.path.join(output_dir, "gps_factor_debug.csv"))
        self.plot_degraded_regions(os.path.join(output_dir, 'degraded_regions'))
        self.compute_statistics(os.path.join(output_dir, 'statistics.txt'))

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate LIO-SAM fusion trajectory against GPS odometry.")
    parser.add_argument("bag_file", help="Trajectory bag recorded by record_trajectory.py")
    parser.add_argument("gnss_status_file", help="GNSS status CSV (or any path; /gnss_degraded from bag preferred)")
    parser.add_argument("output_dir", nargs="?", default="./evaluation_results", help="Output directory")
    parser.add_argument("--params", dest="params_file", default=None, help="params.yaml used for the run (to apply lio_sam.gpsExtrinsicRot)")
    parser.add_argument("--gps-topic", dest="gps_topic", default="/odometry/gps",
                        help="GPS odometry topic inside the trajectory bag (e.g. /odometry/gps_test for hold-out evaluation)")
    args = parser.parse_args()

    gps_R = None
    gps_src = None
    if args.params_file:
        gps_R, msg = _load_gps_extrinsic_rot_from_params(args.params_file)
        if gps_R is None:
            print(f"Warning: --params provided but gpsExtrinsicRot not usable: {msg}")
        else:
            gps_src = os.path.basename(args.params_file)

    evaluator = TrajectoryEvaluator(args.bag_file, args.gnss_status_file, gps_topic=args.gps_topic,
                                    gps_extrinsic_rot=gps_R, gps_extrinsic_source=gps_src)
    evaluator.run_evaluation(args.output_dir)
