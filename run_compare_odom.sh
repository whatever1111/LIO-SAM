#!/bin/bash
set -euo pipefail

echo "========================================"
echo "LIO-SAM Odom Comparison"
echo "========================================"
echo ""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--bag <bag>] [--start <sec>] [--end <sec>] [--duration <sec>] \\
                   [--output-dir <dir>] [--params <yaml>] \\
                   [--ref-topic <topic>] [--est-topic <topic>] [--out-bag <bag>] \\
                   [--gps-topic <topic>] [--no-gps] \\
                   [--t-max-diff <sec>] [--no-align]

说明:
  - 从 bag 中回放指定时间段(默认 60s-260s)，同时运行 LIO-SAM(run_as_localization.launch)。
  - 录制里程计轨迹并用 evo 进行对比:
      ref: bag 内的原始 /odom 会被 remap 到 ref-topic (默认: /odom_ref)
      est: LIO-SAM 输出里程计 (默认: /odom)
      gps: /odometry/gps (默认启用；若 bag 里没有该 topic，会尝试用 lio_sam_fpaOdomConverter 从 /fixposition/fpa/(odometry|odomenu) 生成)

Options:
  --bag <bag>          rosbag 路径 (默认: ~/autodl-tmp/st_chargeroom_0106_hangzhou_2026-01-06-18-32-40.bag)
  --start <sec>        从 bag 开始后第 N 秒开始回放 (默认: 60)
  --end <sec>          回放到 bag 开始后第 N 秒结束 (默认: 260)
  --duration <sec>     回放时长 (与 --end 二选一; 优先使用 --duration)
  --output-dir <dir>   输出目录 (默认: /tmp/odom_comparison)
  --params <yaml>      params.yaml 路径 (默认: \$CATKIN_WS/src/LIO-SAM/config/params.yaml)
  --ref-topic <topic>  参考 odom topic(回放时 /odom remap 到这里) (默认: /odom_ref)
  --est-topic <topic>  估计 odom topic(默认: /odom; 若设置成其它，会用 relay 从 /odom 转发) (默认: /odom)
  --out-bag <bag>      录制输出 bag (默认: /tmp/odom_compare.bag)
  --gps-topic <topic>  GPS odom topic (默认: /odometry/gps)
  --no-gps             不生成/不对比 GPS（只对比 ref vs est）
  --t-max-diff <sec>   evo 轨迹时间戳匹配最大容差 (默认: 0.1)
  --no-align           不做 evo 的 SE3 对齐(默认会用 -a 对齐后给出对比)
EOF
}

CATKIN_WS="/root/autodl-tmp/catkin_ws"
DEFAULT_BAG="${HOME}/autodl-tmp/st_chargeroom_0106_hangzhou_2026-01-06-18-32-40.bag"

BAG_FILE="$DEFAULT_BAG"
PLAY_START="60"
PLAY_END="260"
PLAY_DURATION=""
OUTPUT_DIR="/tmp/odom_comparison"
PARAMS_FILE="${CATKIN_WS}/src/LIO-SAM/config/params.yaml"

REF_TOPIC="/odom_ref"
EST_TOPIC="/odom"
GPS_TOPIC="/odometry/gps"
COMPARE_GPS=true
OUT_BAG="/tmp/odom_compare.bag"
T_MAX_DIFF="0.1"
DO_ALIGN=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      BAG_FILE="$2"
      shift 2
      ;;
    --start)
      PLAY_START="$2"
      shift 2
      ;;
    --end)
      PLAY_END="$2"
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
    --ref-topic)
      REF_TOPIC="$2"
      shift 2
      ;;
    --est-topic)
      EST_TOPIC="$2"
      shift 2
      ;;
    --out-bag)
      OUT_BAG="$2"
      shift 2
      ;;
    --gps-topic)
      GPS_TOPIC="$2"
      shift 2
      ;;
    --no-gps)
      COMPARE_GPS=false
      shift
      ;;
    --t-max-diff)
      T_MAX_DIFF="$2"
      shift 2
      ;;
    --no-align)
      DO_ALIGN=false
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

OUT_BAG="$(python3 - "$OUT_BAG" <<'PY'
import os
import sys
print(os.path.expanduser(sys.argv[1]))
PY
)"

if [[ "$REF_TOPIC" == "$EST_TOPIC" ]]; then
  echo "Error: --ref-topic must be different from --est-topic (got: $REF_TOPIC)"
  exit 2
fi
if [[ "$COMPARE_GPS" == "true" && "$GPS_TOPIC" == "$REF_TOPIC" ]]; then
  echo "Error: --gps-topic must be different from --ref-topic (got: $GPS_TOPIC)"
  exit 2
fi
if [[ "$COMPARE_GPS" == "true" && "$GPS_TOPIC" == "$EST_TOPIC" ]]; then
  echo "Error: --gps-topic must be different from --est-topic (got: $GPS_TOPIC)"
  exit 2
fi

if [[ ! -f "$BAG_FILE" ]]; then
  echo "Error: Bag file not found: $BAG_FILE"
  exit 1
fi

if [[ ! -f "$CATKIN_WS/devel/setup.bash" ]]; then
  echo "Error: catkin workspace not built: $CATKIN_WS/devel/setup.bash not found"
  exit 1
fi

if [[ -n "$PLAY_DURATION" ]]; then
  :
else
  PLAY_DURATION="$(python3 - <<PY
start=float("$PLAY_START")
end=float("$PLAY_END")
if end <= start:
    raise SystemExit(f"--end must be > --start, got end={end} start={start}")
print(end-start)
PY
)"
fi

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
echo "Step 2: Publishing base_link->lidar_link TF (from bag /tf_static, fallback=identity)..."
echo "----------------------------------------"
read -r BL_LIDAR_XYZ BL_LIDAR_QUAT < <(
  python3 - "$BAG_FILE" <<'PY' 2>/dev/null || echo "[0,0,0] [0,0,0,1]"
import os
import rosbag
import sys

bag_path = os.path.expanduser(sys.argv[1])

def norm(s): return (s or "").strip().lstrip("/")

xyz = (0.0, 0.0, 0.0)
quat = (0.0, 0.0, 0.0, 1.0)
found = False
with rosbag.Bag(bag_path, "r") as bag:
    for _topic, msg, _t in bag.read_messages(topics=["/tf_static"]):
        for tr in msg.transforms:
            if norm(tr.header.frame_id) == "base_link" and norm(tr.child_frame_id) == "lidar_link":
                t = tr.transform.translation
                q = tr.transform.rotation
                xyz = (float(t.x), float(t.y), float(t.z))
                quat = (float(q.x), float(q.y), float(q.z), float(q.w))
                found = True
                break
        if found:
            break

xyz_s = f"[{xyz[0]:.6g},{xyz[1]:.6g},{xyz[2]:.6g}]"
q_s = f"[{quat[0]:.6g},{quat[1]:.6g},{quat[2]:.6g},{quat[3]:.6g}]"
print(xyz_s, q_s)
PY
)

# Convert "[x,y,z]" -> "x y z" and "[qx,qy,qz,qw]" -> "qx qy qz qw"
BL_LIDAR_ARGS="$(python3 - <<PY
import re
xyz = "$BL_LIDAR_XYZ"
quat = "$BL_LIDAR_QUAT"
def nums(s):
    s = s.strip().strip("[]")
    return [x for x in re.split(r"[\\s,]+", s) if x]
vals = nums(xyz) + nums(quat)
if len(vals) != 7:
    raise SystemExit(f"Expected 7 numbers for static tf, got {len(vals)}: {vals}")
print(" ".join(vals))
PY
)"

rosrun tf2_ros static_transform_publisher $BL_LIDAR_ARGS base_link lidar_link >/dev/null 2>&1 &
TF_BL_LIDAR_PID=$!
cleanup_pids+=("$TF_BL_LIDAR_PID")
sleep 1

echo ""
echo "Step 3: Starting LIO-SAM (run_as_localization.launch)..."
echo "----------------------------------------"
roslaunch lio_sam run_as_localization.launch params:="$PARAMS_FILE" &
LAUNCH_PID=$!
cleanup_pids+=("$LAUNCH_PID")
sleep 3

echo ""
echo "Step 4: Preparing GPS topic (optional)..."
echo "----------------------------------------"
rm -f "$OUT_BAG" || true

GPS_MODE="disabled"
GPS_INPUT_TOPIC=""
if [[ "$COMPARE_GPS" == "true" ]]; then
  read -r GPS_MODE GPS_INPUT_TOPIC < <(
    python3 - "$BAG_FILE" "$GPS_TOPIC" <<'PY'
import os
import sys
import rosbag

bag_path = os.path.expanduser(sys.argv[1])
gps_topic = sys.argv[2]

def has_topic(bag, topic):
    try:
        info = bag.get_type_and_topic_info()[1]
        return topic in info and info[topic].message_count > 0
    except Exception:
        return False

with rosbag.Bag(bag_path, "r") as bag:
    if has_topic(bag, gps_topic):
        print("bag", gps_topic)
    elif has_topic(bag, "/fixposition/fpa/odometry"):
        print("converter", "/fixposition/fpa/odometry")
    elif has_topic(bag, "/fixposition/fpa/odomenu"):
        print("converter", "/fixposition/fpa/odomenu")
    else:
        print("missing", "")
PY
  )

  if [[ "$GPS_MODE" == "missing" ]]; then
    echo "Warn: cannot compare GPS - bag has neither $GPS_TOPIC nor /fixposition/fpa/(odometry|odomenu)."
    COMPARE_GPS=false
  elif [[ "$GPS_MODE" == "bag" ]]; then
    echo "GPS source: bag topic $GPS_TOPIC"
  elif [[ "$GPS_MODE" == "converter" ]]; then
    echo "GPS source: lio_sam_fpaOdomConverter from $GPS_INPUT_TOPIC -> $GPS_TOPIC"
    if [[ "$GPS_INPUT_TOPIC" == "/fixposition/fpa/odomenu" ]]; then
      rosrun lio_sam lio_sam_fpaOdomConverter __name:=fpa_odom_converter_compare _input_type:=odomenu _input_topic:="$GPS_INPUT_TOPIC" _output_topic:="$GPS_TOPIC" _use_receive_time:=true _zero_initial_position:=true >/dev/null 2>&1 &
    else
      rosrun lio_sam lio_sam_fpaOdomConverter __name:=fpa_odom_converter_compare _input_type:=odometry _input_topic:="$GPS_INPUT_TOPIC" _output_topic:="$GPS_TOPIC" _use_receive_time:=true _zero_initial_position:=true >/dev/null 2>&1 &
    fi
    GPS_CONV_PID=$!
    cleanup_pids+=("$GPS_CONV_PID")
    sleep 1
  else
    echo "Warn: unexpected GPS_MODE=$GPS_MODE (skip GPS comparison)."
    COMPARE_GPS=false
  fi
fi

echo ""
echo "Step 5: Starting odom recorder..."
echo "----------------------------------------"
if [[ "$EST_TOPIC" != "/odom" ]]; then
  echo "Start relay: /odom -> $EST_TOPIC"
  rosrun topic_tools relay /odom "$EST_TOPIC" >/dev/null 2>&1 &
  RELAY_PID=$!
  cleanup_pids+=("$RELAY_PID")
  sleep 1
fi

REC_TOPICS=("$REF_TOPIC" "$EST_TOPIC")
if [[ "$COMPARE_GPS" == "true" ]]; then
  REC_TOPICS+=("$GPS_TOPIC")
fi

echo "Recording: ${REC_TOPICS[*]}"
rosbag record -O "$OUT_BAG" "${REC_TOPICS[@]}" >/dev/null 2>&1 &
REC_PID=$!
cleanup_pids+=("$REC_PID")
sleep 1

PLAY_TOPICS=(/lidar_points /imu/data /odom)
PLAY_REMAPS=("/odom:=$REF_TOPIC")
if [[ "$COMPARE_GPS" == "true" ]]; then
  if [[ "$GPS_MODE" == "bag" ]]; then
    PLAY_TOPICS+=("$GPS_TOPIC")
  elif [[ "$GPS_MODE" == "converter" ]]; then
    PLAY_TOPICS+=("$GPS_INPUT_TOPIC")
  fi
fi

PLAY_ARGS=(--clock --delay=3 --start "$PLAY_START" --duration "$PLAY_DURATION" --topics "${PLAY_TOPICS[@]}" "${PLAY_REMAPS[@]}")

echo ""
echo "Step 6: Playing bag (topics only)..."
echo "----------------------------------------"
echo "rosbag play $BAG_FILE ${PLAY_ARGS[*]}"
rosbag play "$BAG_FILE" "${PLAY_ARGS[@]}" &
BAG_PID=$!
cleanup_pids+=("$BAG_PID")

echo ""
echo "Step 7: Waiting for bag playback to finish..."
echo "----------------------------------------"
wait "$BAG_PID" 2>/dev/null || true

echo ""
echo "Step 8: Stopping recorder..."
echo "----------------------------------------"
kill -2 "$REC_PID" 2>/dev/null || true
wait "$REC_PID" 2>/dev/null || true
sleep 1

echo ""
echo "Step 9: Stopping LIO-SAM..."
echo "----------------------------------------"
kill -2 "$LAUNCH_PID" 2>/dev/null || true
wait "$LAUNCH_PID" 2>/dev/null || true
sleep 1

echo ""
echo "Step 10: Running evo comparison..."
echo "----------------------------------------"
mkdir -p "$OUTPUT_DIR"

echo "Sanity check: ensure required topics exist in $OUT_BAG..."
python3 - "$OUT_BAG" "$REF_TOPIC" "$EST_TOPIC" "$COMPARE_GPS" "$GPS_TOPIC" <<'PY'
import sys
import rosbag

bag_path = sys.argv[1]
ref_topic = sys.argv[2]
est_topic = sys.argv[3]
compare_gps = sys.argv[4].lower() == "true"
gps_topic = sys.argv[5]

info = rosbag.Bag(bag_path, "r", allow_unindexed=True).get_type_and_topic_info()[1]
ref_cnt = info.get(ref_topic).message_count if ref_topic in info else 0
est_cnt = info.get(est_topic).message_count if est_topic in info else 0

print(f"  {ref_topic}: {ref_cnt}")
print(f"  {est_topic}: {est_cnt}")
if compare_gps:
    gps_cnt = info.get(gps_topic).message_count if gps_topic in info else 0
    print(f"  {gps_topic}: {gps_cnt}")
else:
    gps_cnt = 1

if ref_cnt <= 0 or est_cnt <= 0 or gps_cnt <= 0:
    print("Error: recorded bag missing required topics. Available topics:")
    for t, ti in sorted(info.items()):
        print(f"  - {t} ({ti.msg_type}) count={ti.message_count}")
    raise SystemExit(1)
PY

ALIGN_FLAG=()
if [[ "$DO_ALIGN" == "true" ]]; then
  ALIGN_FLAG=(-a)
fi

cd "$OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Recorded bag: $OUT_BAG"

# ref vs est
evo_traj bag "$OUT_BAG" "$REF_TOPIC" "$EST_TOPIC" --ref "$REF_TOPIC" --sync --t_max_diff "$T_MAX_DIFF" "${ALIGN_FLAG[@]}" --plot_mode xy --save_plot traj_xy_ref_est.png --save_as_tum --no_warnings >traj_xy_ref_est.log 2>&1 || true
evo_ape  bag "$OUT_BAG" "$REF_TOPIC" "$EST_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" --no_warnings >ape_trans_ref_est_raw.txt 2>&1 || true
evo_ape  bag "$OUT_BAG" "$REF_TOPIC" "$EST_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" "${ALIGN_FLAG[@]}" --save_plot ape_trans_ref_est.png --save_results ape_trans_ref_est.zip --no_warnings >ape_trans_ref_est.txt 2>&1 || true
evo_rpe  bag "$OUT_BAG" "$REF_TOPIC" "$EST_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" "${ALIGN_FLAG[@]}" --save_plot rpe_trans_ref_est.png --save_results rpe_trans_ref_est.zip --no_warnings >rpe_trans_ref_est.txt 2>&1 || true

if [[ "$COMPARE_GPS" == "true" ]]; then
  # gps vs est
  evo_traj bag "$OUT_BAG" "$GPS_TOPIC" "$EST_TOPIC" --ref "$GPS_TOPIC" --sync --t_max_diff "$T_MAX_DIFF" "${ALIGN_FLAG[@]}" --plot_mode xy --save_plot traj_xy_gps_est.png --no_warnings >traj_xy_gps_est.log 2>&1 || true
  evo_ape  bag "$OUT_BAG" "$GPS_TOPIC" "$EST_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" --no_warnings >ape_trans_gps_est_raw.txt 2>&1 || true
  evo_ape  bag "$OUT_BAG" "$GPS_TOPIC" "$EST_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" "${ALIGN_FLAG[@]}" --save_plot ape_trans_gps_est.png --save_results ape_trans_gps_est.zip --no_warnings >ape_trans_gps_est.txt 2>&1 || true
  evo_rpe  bag "$OUT_BAG" "$GPS_TOPIC" "$EST_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" "${ALIGN_FLAG[@]}" --save_plot rpe_trans_gps_est.png --save_results rpe_trans_gps_est.zip --no_warnings >rpe_trans_gps_est.txt 2>&1 || true

  # gps vs ref
  evo_traj bag "$OUT_BAG" "$GPS_TOPIC" "$REF_TOPIC" --ref "$GPS_TOPIC" --sync --t_max_diff "$T_MAX_DIFF" --plot_mode xy --save_plot traj_xy_gps_ref.png --no_warnings >traj_xy_gps_ref.log 2>&1 || true
  evo_ape  bag "$OUT_BAG" "$GPS_TOPIC" "$REF_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" --no_warnings >ape_trans_gps_ref_raw.txt 2>&1 || true
  evo_ape  bag "$OUT_BAG" "$GPS_TOPIC" "$REF_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" -a --save_plot ape_trans_gps_ref.png --save_results ape_trans_gps_ref.zip --no_warnings >ape_trans_gps_ref.txt 2>&1 || true
  evo_rpe  bag "$OUT_BAG" "$GPS_TOPIC" "$REF_TOPIC" -r trans_part --t_max_diff "$T_MAX_DIFF" -a --save_plot rpe_trans_gps_ref.png --save_results rpe_trans_gps_ref.zip --no_warnings >rpe_trans_gps_ref.txt 2>&1 || true
fi

echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo "Results: $OUTPUT_DIR"
echo "Recorded odom bag: $OUT_BAG"
echo " - ref vs est: traj_xy_ref_est_* / ape_trans_ref_est_* / rpe_trans_ref_est_*"
if [[ "$COMPARE_GPS" == "true" ]]; then
  echo " - gps vs est: traj_xy_gps_est_* / ape_trans_gps_est_* / rpe_trans_gps_est_*"
  echo " - gps vs ref: traj_xy_gps_ref_* / ape_trans_gps_ref_* / rpe_trans_gps_ref_*"
fi
echo ""
