#!/bin/bash
set -euo pipefail

# Evaluation helper script:
# - Verifies TF availability (static + dynamic) required by LIO-SAM.
# - Optionally replays a bag for a fixed duration and runs analysis tools.
# - Designed to work with the updated pipeline where mapping degeneracy is published as
#   /lio_sam/mapping/odometry_incremental_status (MappingStatus) instead of reusing covariance[0].

echo "========================================"
echo "LIO-SAM Evaluation / TF Verification"
echo "========================================"
echo ""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--bag <bag>] [--duration <sec>] [--tf-only] [--verify-imu] [--rep103]

Options:
  --bag <bag>        rosbag 路径 (默认: ~/autodl-tmp/st_chargeroom_1222_2025-12-22-12-16-41.bag)
  --duration <sec>   只回放前 N 秒 (bag-time)
  --tf-only          仅构建/验证所需 TF，不跑评估绘图
  --verify-imu       离线输出 RAWIMU/IMU/ODOMENU 对齐报告后退出
  --rep103           运行时通过 TF+点云重发布将 LiDAR 修正到 REP-103 (lidar_link_rep103)
  --reuse-master     复用当前 ROS_MASTER_URI (默认会启动隔离 roscore，避免已有节点/TF 干扰)
EOF
}

CATKIN_WS="/root/autodl-tmp/catkin_ws"
DEFAULT_BAG="${HOME}/autodl-tmp/st_chargeroom_1222_2025-12-22-12-16-41.bag"

BAG_FILE="$DEFAULT_BAG"
PLAY_DURATION=""
TF_ONLY=false
VERIFY_IMU=false
USE_REP103=false
REUSE_MASTER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      BAG_FILE="$2"
      shift 2
      ;;
    --duration)
      PLAY_DURATION="$2"
      shift 2
      ;;
    --tf-only)
      TF_ONLY=true
      shift
      ;;
    --verify|--verify-imu)
      VERIFY_IMU=true
      shift
      ;;
    --rep103|--lidar-rep103)
      USE_REP103=true
      shift
      ;;
    --reuse-master)
      REUSE_MASTER=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

BAG_FILE="$(python3 - <<PY
import os
print(os.path.expanduser("${BAG_FILE}"))
PY
)"

TRAJ_BAG="/tmp/trajectory_evaluation.bag"
GNSS_STATUS="/tmp/gnss_status.txt"
OUTPUT_DIR="/tmp/evaluation_results"
FPA_ALIGN_JSON="/tmp/fpa_rawimu_alignment.json"

if [[ ! -f "$BAG_FILE" ]]; then
  echo "Error: Bag file not found: $BAG_FILE"
  exit 1
fi

if [[ ! -f "$CATKIN_WS/devel/setup.bash" ]]; then
  echo "Error: catkin workspace not built: $CATKIN_WS/devel/setup.bash not found"
  exit 1
fi

# Make ROS cache/log writable in restricted environments
export ROS_HOME="${ROS_HOME:-/tmp/ros}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$ROS_HOME/log}"
export ROS_IP="${ROS_IP:-127.0.0.1}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR"

cd "$CATKIN_WS"
source devel/setup.bash

echo "Bag file: $BAG_FILE"
echo ""

echo "Step 0: Offline TF requirement report (from bag)..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/report_bag_tf.py --bag "$BAG_FILE" --params src/LIO-SAM/config/params.yaml || true

echo ""
echo "Step 0b: Offline FPA RAWIMU alignment report (from bag)..."
echo "----------------------------------------"
rm -f "$FPA_ALIGN_JSON" || true
python3 src/LIO-SAM/scripts/report_fpa_rawimu_alignment.py --bag "$BAG_FILE" --params src/LIO-SAM/config/params.yaml --export-json "$FPA_ALIGN_JSON" || true

if [[ "$VERIFY_IMU" == "true" ]]; then
  echo ""
  echo "verify-imu mode: offline report complete."
  exit 0
fi

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    kill -2 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo ""
echo "Step 1: Starting roscore..."
echo "----------------------------------------"
if [[ "$REUSE_MASTER" == "true" ]]; then
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
  if ! timeout 2 rosparam list >/dev/null 2>&1; then
    roscore >/tmp/roscore.log 2>&1 &
    ROSCORE_PID=$!
    cleanup_pids+=("$ROSCORE_PID")
    sleep 2
  fi
else
  # Always start a private master to avoid interference from other running ROS nodes/TF publishers.
  ROS_MASTER_PORT="$(
    python3 - <<'PY'
import socket
s=socket.socket()
s.bind(('127.0.0.1',0))
print(s.getsockname()[1])
s.close()
PY
  )"
  export ROS_MASTER_URI="http://127.0.0.1:${ROS_MASTER_PORT}"
  roscore -p "${ROS_MASTER_PORT}" >/tmp/roscore.log 2>&1 &
  ROSCORE_PID=$!
  cleanup_pids+=("$ROSCORE_PID")
  sleep 2
fi
echo "ROS_MASTER_URI=$ROS_MASTER_URI"
rosparam set /use_sim_time true >/dev/null 2>&1 || true

echo ""
echo "Step 2: Preparing required static TF publishers..."
echo "----------------------------------------"
POI_VRTK_XYZ="[0,0,0]"
POI_VRTK_RPY_DEG="[0,0,0]"
if read -r POI_VRTK_XYZ POI_VRTK_RPY_DEG < <(
  python3 - <<PY
import os
import math
import rosbag

bag_path = os.path.expanduser("${BAG_FILE}")

def norm(s): return (s or "").strip().lstrip("/")

def quat_to_rpy(x, y, z, w):
    # ROS: xyzw
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

xyz = (0.0, 0.0, 0.0)
rpy_deg = (0.0, 0.0, 0.0)

found = False
with rosbag.Bag(bag_path, "r") as bag:
    for _topic, msg, _t in bag.read_messages(topics=["/tf_static"]):
        for tr in msg.transforms:
            if norm(tr.header.frame_id) == "FP_POI" and norm(tr.child_frame_id) == "FP_VRTK":
                t = tr.transform.translation
                q = tr.transform.rotation
                xyz = (float(t.x), float(t.y), float(t.z))
                r, p, y = quat_to_rpy(float(q.x), float(q.y), float(q.z), float(q.w))
                rpy_deg = (math.degrees(r), math.degrees(p), math.degrees(y))
                found = True
                break
        if found:
            break

xyz_s = f"[{xyz[0]:.6g},{xyz[1]:.6g},{xyz[2]:.6g}]"
rpy_s = f"[{rpy_deg[0]:.6g},{rpy_deg[1]:.6g},{rpy_deg[2]:.6g}]"
print(xyz_s, rpy_s)
PY
); then
  :
else
  echo "Warn: failed to extract FP_POI->FP_VRTK from bag /tf_static; fallback to identity."
  POI_VRTK_XYZ="[0,0,0]"
  POI_VRTK_RPY_DEG="[0,0,0]"
fi

if [[ "$USE_REP103" == "true" ]]; then
  echo "REP103 mode: publish base_link->lidar_link via static_transform_publisher (for chaining to lidar_link_rep103)"
  rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 base_link lidar_link >/dev/null 2>&1 &
  TF_LIDAR_PID=$!
  cleanup_pids+=("$TF_LIDAR_PID")
  sleep 1
fi

if [[ "$USE_REP103" == "true" ]]; then
  echo ""
  echo "Step 2b: Starting LiDAR REP-103 republisher (lidar_link -> lidar_link_rep103)..."
  echo "----------------------------------------"
  python3 src/LIO-SAM/scripts/republish_lidar_tf.py &
  REP103_PID=$!
  cleanup_pids+=("$REP103_PID")
  sleep 1
fi

echo ""
echo "Step 3: Starting LIO-SAM (run_evaluation.launch)..."
echo "----------------------------------------"
TMP_PARAMS="/tmp/lio_sam_params_eval.yaml"
# Build a temporary params file:
# - Optionally apply gpsExtrinsicRot derived from dual-IMU yaw offset (offline JSON).
# - Optionally apply REP-103 LiDAR frame correction (Rz(pi)) keeping TF + params consistent.
USE_REP103_ENV=0
if [[ "$USE_REP103" == "true" ]]; then
  USE_REP103_ENV=1
fi
python3 - <<PY
import math
import os
import json
import yaml

src = os.path.join("$CATKIN_WS", "src", "LIO-SAM", "config", "params.yaml")
dst = "$TMP_PARAMS"
align_json = "$FPA_ALIGN_JSON"
use_rep103 = bool(int("$USE_REP103_ENV"))

with open(src, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

lio = data.get("lio_sam", {})

def mat3(key):
    v = lio.get(key)
    if not isinstance(v, list) or len(v) != 9:
        return None
    return [float(x) for x in v]

def vec3(key):
    v = lio.get(key)
    if not isinstance(v, list) or len(v) != 3:
        raise SystemExit(f"Missing/invalid {key} in params.yaml")
    return [float(x) for x in v]

def mm(A, B):
    # 3x3 row-major multiply
    out = [0.0] * 9
    for r in range(3):
        for c in range(3):
            out[r*3+c] = sum(A[r*3+k] * B[k*3+c] for k in range(3))
    return out

def mv(A, v):
    return [
        A[0]*v[0] + A[1]*v[1] + A[2]*v[2],
        A[3]*v[0] + A[4]*v[1] + A[5]*v[2],
        A[6]*v[0] + A[7]*v[1] + A[8]*v[2],
    ]

def set_if_valid_mat3(key, value):
    if isinstance(value, list) and len(value) == 9:
        lio[key] = [float(x) for x in value]

def read_align():
    if not os.path.exists(align_json):
        return None
    try:
        with open(align_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

align = read_align()
if isinstance(align, dict):
    rec = align.get("recommended_gps_extrinsic_rot_rowmajor")
    sd = align.get("yaw_diff_circ_std_deg")
    matched = align.get("yaw_diff_matched")
    gyro = align.get("gyro_fit_R_imu_link_from_FP_VRTK_rowmajor")
    gyro_ok = True
    if isinstance(gyro, list) and len(gyro) == 9:
        tr = float(gyro[0]) + float(gyro[4]) + float(gyro[8])
        v = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        ang_deg = math.degrees(math.acos(v))
        gyro_ok = ang_deg <= 10.0
        if not gyro_ok:
            print("Skip auto gpsExtrinsicRot (gyro misaligned): angle_deg=", ang_deg)
    if gyro_ok and isinstance(rec, list) and len(rec) == 9 and (sd is None or float(sd) < 5.0) and (matched is None or int(matched) >= 200):
        set_if_valid_mat3("gpsExtrinsicRot", rec)
        print("Applied gpsExtrinsicRot from alignment json; yaw_deg=", align.get("recommended_gps_extrinsic_yaw_deg"))
    else:
        print("Skip auto gpsExtrinsicRot (rec/sd/matched invalid):", {"sd": sd, "matched": matched})
else:
    print("No alignment json, keep gpsExtrinsicRot from params.yaml")

Rz_pi = [-1.0, 0.0, 0.0,
         0.0,-1.0, 0.0,
         0.0, 0.0, 1.0]

lio["publishBaseToLidarTf"] = (not use_rep103)  # REP103 uses base->lidar_link + lidar_link->lidar_link_rep103 chain

if use_rep103:
    lio["pointCloudTopic"] = "/lidar_points_rep103"
    lio["lidarFrame"] = "lidar_link_rep103"

    # Rotate all LiDAR-frame-defined extrinsics into the REP-103 LiDAR frame.
    R = mat3("extrinsicRot")
    if R is not None:
        lio["extrinsicRot"] = mm(Rz_pi, R)
    Rrpy = mat3("extrinsicRPY")
    if Rrpy is not None:
        lio["extrinsicRPY"] = mm(Rz_pi, Rrpy)
    G = mat3("gpsExtrinsicRot")
    if G is not None:
        lio["gpsExtrinsicRot"] = mm(Rz_pi, G)
    lio["extrinsicTrans"] = mv(Rz_pi, vec3("extrinsicTrans"))

data["lio_sam"] = lio
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("Wrote", dst)
PY
if [[ "$USE_REP103" == "true" ]]; then
  roslaunch lio_sam run_evaluation.launch params_file:="$TMP_PARAMS" pointCloudTopic:=/lidar_points_rep103 lidarFrame:=lidar_link_rep103 publish_poi_to_vrtk_tf:=true poi_to_vrtk_xyz:="$POI_VRTK_XYZ" poi_to_vrtk_rpy_deg:="$POI_VRTK_RPY_DEG" &
else
  if [[ "$TF_ONLY" == "true" ]]; then
    roslaunch lio_sam run_tf_only.launch params_file:="$TMP_PARAMS" publish_poi_to_vrtk_tf:=true poi_to_vrtk_xyz:="$POI_VRTK_XYZ" poi_to_vrtk_rpy_deg:="$POI_VRTK_RPY_DEG" &
  else
    roslaunch lio_sam run_evaluation.launch params_file:="$TMP_PARAMS" publish_poi_to_vrtk_tf:=true poi_to_vrtk_xyz:="$POI_VRTK_XYZ" poi_to_vrtk_rpy_deg:="$POI_VRTK_RPY_DEG" &
  fi
fi
LAUNCH_PID=$!
cleanup_pids+=("$LAUNCH_PID")
sleep 3

echo ""
echo "Step 4: Runtime TF verification..."
echo "----------------------------------------"
LIDAR_FRAME="lidar_link"
TF_IDENTITY_FLAG=()
if [[ "$USE_REP103" == "true" ]]; then
  LIDAR_FRAME="lidar_link_rep103"
  TF_IDENTITY_FLAG=(--no-base-lidar-identity)
fi
TF_VERIFY_EXTRA=()
if [[ "$TF_ONLY" == "true" ]]; then
  # run_tf_only.launch publishes map->odom and odom->base_link statically for verification
  TF_VERIFY_EXTRA+=(--identity-trans-tol 0.02 --identity-rot-tol-deg 1.0)
fi
python3 src/LIO-SAM/scripts/verify_required_tf.py --timeout 60 --map map --odom odom --base base_link --lidar "$LIDAR_FRAME" "${TF_IDENTITY_FLAG[@]}" "${TF_VERIFY_EXTRA[@]}" &
TF_VERIFY_PID=$!
cleanup_pids+=("$TF_VERIFY_PID")

echo ""
echo "Step 4b: TF vs params consistency check..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/verify_params_tf_consistency.py --timeout 60 --trans-tol 1e-3 --rot-tol-deg 0.5 &
TF_CONSIST_PID=$!
cleanup_pids+=("$TF_CONSIST_PID")

echo ""
echo "Step 4c: Verify RTK(FP_*) TF is connected to robot TF tree..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/verify_rtk_tf_connection.py --timeout 90 --base base_link &
TF_RTK_PID=$!
cleanup_pids+=("$TF_RTK_PID")

echo ""
echo "Step 4d: Verify FP_ENU0->map TF matches gpsExtrinsicRot..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/verify_fpa_map_tf_consistency.py --timeout 90 --enu FP_ENU0 --map map --rot-tol-deg 1.0 &
TF_FPA_MAP_PID=$!
cleanup_pids+=("$TF_FPA_MAP_PID")

if [[ "$TF_ONLY" == "true" ]]; then
  if [[ -z "$PLAY_DURATION" ]]; then
    PLAY_DURATION="60"
  fi
fi

PLAY_ARGS=(--clock --delay=3 --topics /lidar_points /imu/data /fixposition/fpa/rawimu /fixposition/fpa/imubias /fixposition/fpa/odometry /fixposition/fpa/odomenu)
if [[ -n "$PLAY_DURATION" ]]; then
  PLAY_ARGS+=(--duration "$PLAY_DURATION")
fi

echo ""
if [[ "$TF_ONLY" != "true" ]]; then
  echo "Step 5: Starting trajectory recorder..."
  echo "----------------------------------------"
  python3 src/LIO-SAM/scripts/record_trajectory.py "$TRAJ_BAG" &
  RECORDER_PID=$!
  cleanup_pids+=("$RECORDER_PID")
  sleep 1
fi

echo ""
echo "Step 6: Playing bag (topics only)..."
echo "----------------------------------------"
echo "rosbag play $BAG_FILE ${PLAY_ARGS[*]}"
rosbag play "$BAG_FILE" "${PLAY_ARGS[@]}" &
BAG_PID=$!
cleanup_pids+=("$BAG_PID")

	set +e
	wait "$TF_VERIFY_PID"
	TF_VERIFY_RC=$?
	wait "$TF_CONSIST_PID"
	TF_CONSIST_RC=$?
		wait "$TF_RTK_PID"
		TF_RTK_RC=$?
		wait "$TF_FPA_MAP_PID"
		TF_FPA_MAP_RC=$?
		set -e
	if [[ $TF_VERIFY_RC -ne 0 ]]; then
	  echo "Error: TF verification failed (rc=$TF_VERIFY_RC)."
	  exit $TF_VERIFY_RC
	fi
	if [[ $TF_CONSIST_RC -ne 0 ]]; then
	  echo "Error: TF vs params consistency check failed (rc=$TF_CONSIST_RC)."
	  exit $TF_CONSIST_RC
	fi
		if [[ $TF_RTK_RC -ne 0 ]]; then
		  echo "Error: RTK TF connectivity check failed (rc=$TF_RTK_RC)."
		  exit $TF_RTK_RC
		fi
		if [[ $TF_FPA_MAP_RC -ne 0 ]]; then
		  echo "Error: FP_ENU0->map TF consistency check failed (rc=$TF_FPA_MAP_RC)."
		  exit $TF_FPA_MAP_RC
		fi

if [[ "$TF_ONLY" == "true" ]]; then
  echo ""
  echo "TF verification complete (tf-only)."
  exit 0
fi

echo ""
echo "Step 7: Waiting for bag playback to finish..."
echo "----------------------------------------"
wait "$BAG_PID" 2>/dev/null || true

echo ""
echo "Step 8: Stopping recorder..."
echo "----------------------------------------"
kill -2 "$RECORDER_PID" 2>/dev/null || true
wait "$RECORDER_PID" 2>/dev/null || true
sleep 1

echo ""
echo "Step 9: Stopping LIO-SAM..."
echo "----------------------------------------"
kill -2 "$LAUNCH_PID" 2>/dev/null || true
wait "$LAUNCH_PID" 2>/dev/null || true
sleep 1

echo ""
echo "Step 10: Running evaluation and generating plots..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/evaluate_trajectory.py "$TRAJ_BAG" "$GNSS_STATUS" "$OUTPUT_DIR" --params src/LIO-SAM/config/params.yaml

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results: $OUTPUT_DIR"
echo "GNSS status log: $GNSS_STATUS"
echo "Trajectory bag: $TRAJ_BAG"
echo ""
