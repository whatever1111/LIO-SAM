#!/bin/bash
set -euo pipefail

echo "========================================"
echo "LIO-SAM Evaluation (Pure)"
echo "========================================"
echo ""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--bag <bag>] [--duration <sec>] [--output-dir <dir>] [--params <yaml>]

Options:
  --bag <bag>          rosbag 路径 (默认: ~/autodl-tmp/st_chargeroom_1222_2025-12-22-12-16-41.bag)
  --duration <sec>     只回放前 N 秒 (bag-time)
  --output-dir <dir>   评估输出目录 (默认: /tmp/evaluation_results)
  --params <yaml>      params.yaml 路径 (默认: \$CATKIN_WS/src/LIO-SAM/config/params.yaml)
EOF
}

CATKIN_WS="/root/autodl-tmp/catkin_ws"
DEFAULT_BAG="${HOME}/autodl-tmp/st_chargeroom_1222_2025-12-22-12-16-41.bag"

BAG_FILE="$DEFAULT_BAG"
PLAY_DURATION=""
OUTPUT_DIR="/tmp/evaluation_results"
PARAMS_FILE="${CATKIN_WS}/src/LIO-SAM/config/params.yaml"

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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --params)
      PARAMS_FILE="$2"
      shift 2
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

BAG_FILE="$(python3 - "$BAG_FILE" <<'PY'
import os
import sys
print(os.path.expanduser(sys.argv[1]))
PY
)"

PARAMS_FILE="$(python3 - "$PARAMS_FILE" <<'PY'
import os
import sys
print(os.path.expanduser(sys.argv[1]))
PY
)"

OUTPUT_DIR="$(python3 - "$OUTPUT_DIR" <<'PY'
import os
import sys
print(os.path.expanduser(sys.argv[1]))
PY
)"

TRAJ_BAG="/tmp/trajectory_evaluation.bag"
GNSS_STATUS="/tmp/gnss_status.txt"

# Make ROS cache/log writable in restricted environments
export ROS_HOME="${ROS_HOME:-/tmp/ros}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$ROS_HOME/log}"
export ROS_IP="${ROS_IP:-127.0.0.1}"
mkdir -p "$ROS_HOME" "$ROS_LOG_DIR" "$OUTPUT_DIR"

cd "$CATKIN_WS"
source devel/setup.bash

cleanup_pids=()
cleanup() {
  for pid in "${cleanup_pids[@]:-}"; do
    kill -2 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "Step 1: Using existing roscore..."
echo "----------------------------------------"
if ! timeout 2 rosparam list >/dev/null 2>&1; then
  echo "Error: roscore not reachable. Please start a single global roscore first."
  echo "ROS_MASTER_URI=${ROS_MASTER_URI:-<unset>}"
  exit 1
fi
rosparam set /use_sim_time true >/dev/null 2>&1 || true

echo ""
echo "Step 2: Preparing POI->VRTK TF (from bag /tf_static, fallback=identity)..."
echo "----------------------------------------"
read -r POI_VRTK_XYZ POI_VRTK_RPY_DEG < <(
  python3 - "$BAG_FILE" <<'PY' 2>/dev/null || echo "[0,0,0] [0,0,0]"
import os
import math
import rosbag
import sys

bag_path = os.path.expanduser(sys.argv[1])

def norm(s): return (s or "").strip().lstrip("/")

def quat_to_rpy(x, y, z, w):
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
)

echo ""
echo "Step 3: Starting LIO-SAM (run_evaluation.launch)..."
echo "----------------------------------------"
roslaunch lio_sam run_evaluation.launch params_file:="$PARAMS_FILE" publish_poi_to_vrtk_tf:=true poi_to_vrtk_xyz:="$POI_VRTK_XYZ" poi_to_vrtk_rpy_deg:="$POI_VRTK_RPY_DEG" &
LAUNCH_PID=$!
cleanup_pids+=("$LAUNCH_PID")
sleep 3

echo ""
echo "Step 4: Starting trajectory recorder..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/record_trajectory.py "$TRAJ_BAG" &
RECORDER_PID=$!
cleanup_pids+=("$RECORDER_PID")
sleep 1

PLAY_ARGS=(--clock --delay=3 -s 60  --topics /lidar_points /imu/data /fixposition/fpa/rawimu /fixposition/fpa/imubias /fixposition/fpa/odometry /fixposition/fpa/odomenu)
if [[ -n "$PLAY_DURATION" ]]; then
  PLAY_ARGS+=(--duration "$PLAY_DURATION")

fi

echo ""
echo "Step 5: Playing bag (topics only)..."
echo "----------------------------------------"
echo "rosbag play $BAG_FILE ${PLAY_ARGS[*]}"
rosbag play "$BAG_FILE" "${PLAY_ARGS[@]}" &
BAG_PID=$!
cleanup_pids+=("$BAG_PID")

echo ""
echo "Step 6: Waiting for bag playback to finish..."
echo "----------------------------------------"
wait "$BAG_PID" 2>/dev/null || true

echo ""
echo "Step 7: Stopping recorder..."
echo "----------------------------------------"
kill -2 "$RECORDER_PID" 2>/dev/null || true
wait "$RECORDER_PID" 2>/dev/null || true
sleep 1

echo ""
echo "Step 8: Stopping LIO-SAM..."
echo "----------------------------------------"
kill -2 "$LAUNCH_PID" 2>/dev/null || true
wait "$LAUNCH_PID" 2>/dev/null || true
sleep 1

echo ""
echo "Step 9: Running evaluation and generating plots..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/evaluate_trajectory.py "$TRAJ_BAG" "$GNSS_STATUS" "$OUTPUT_DIR" --params "$PARAMS_FILE"

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results: $OUTPUT_DIR"
echo "GNSS status log: $GNSS_STATUS"
echo "Trajectory bag: $TRAJ_BAG"
echo ""
