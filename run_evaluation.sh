#!/bin/bash

echo "========================================"
echo "LIO-SAM GNSS Fusion Evaluation Pipeline"
echo "========================================"
echo ""

# 解析命令行参数
USE_FPA=false
VERIFY_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --fpa)
            USE_FPA=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$USE_FPA" = true ]; then
    echo "模式: FpaImu (CORRIMU)"
else
    echo "模式: 标准 IMU"
fi
echo ""

# Set use_sim_time for bag file playback
rosparam set /use_sim_time true
# Set paths
CATKIN_WS="/root/autodl-tmp/catkin_ws"
BAG_FILE="/root/autodl-tmp/info_fixed.bag"
TRAJ_BAG="/tmp/trajectory_evaluation.bag"
GNSS_STATUS="/tmp/gnss_status.txt"
OUTPUT_DIR="/tmp/evaluation_results"

# Check if bag file exists
if [ ! -f "$BAG_FILE" ]; then
    echo "Error: Bag file not found: $BAG_FILE"
    exit 1
fi

# Source workspace
cd $CATKIN_WS
source devel/setup.bash

# 如果只是验证模式
if [ "$VERIFY_ONLY" = true ]; then
    echo "启动外参验证模式..."
    echo "----------------------------------------"
    if [ "$USE_FPA" = true ]; then
        python3 src/LIO-SAM/scripts/verify_imu_extrinsics.py --fpa &
    else
        python3 src/LIO-SAM/scripts/verify_imu_extrinsics.py &
    fi
    VERIFY_PID=$!
    sleep 2

    echo "播放bag文件..."
    rosbag play $BAG_FILE --clock --duration=60

    echo "等待验证完成..."
    sleep 10
    kill -2 $VERIFY_PID 2>/dev/null
    exit 0
fi

echo "Step 1: Starting LIO-SAM with GNSS monitor..."
echo "----------------------------------------"
roslaunch lio_sam run_evaluation.launch &
LAUNCH_PID=$!
sleep 5

echo ""
echo "Step 2: Starting trajectory recorder..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/record_trajectory.py $TRAJ_BAG &
RECORDER_PID=$!
sleep 2

echo ""
echo "Step 3: Playing bag file..."
echo "----------------------------------------"
echo "Bag file: $BAG_FILE"
# Use --delay to wait before publishing, giving nodes time to initialize with sim time
rosbag play $BAG_FILE  --clock --delay=3

echo ""
echo "Step 4: Stopping recorder..."
echo "----------------------------------------"
sleep 2
# Send SIGINT (Ctrl+C) to allow graceful shutdown and proper bag closing
kill -2 $RECORDER_PID
# Wait for the process to finish properly
wait $RECORDER_PID 2>/dev/null
sleep 1

echo ""
echo "Step 5: Stopping LIO-SAM..."
echo "----------------------------------------"
kill -2 $LAUNCH_PID
wait $LAUNCH_PID 2>/dev/null
sleep 1

echo ""
echo "Step 6: Running evaluation and generating plots..."
echo "----------------------------------------"
python3 src/LIO-SAM/scripts/evaluate_trajectory.py $TRAJ_BAG $GNSS_STATUS $OUTPUT_DIR

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - trajectory_3d.png: 3D trajectory comparison"
echo "  - trajectory_2d.png: 2D top-down view"
echo "  - errors.png: Error analysis plots"
echo "  - degraded_regions/: Detailed plots for each degraded region"
echo "  - statistics.txt: Numerical error statistics"
echo ""
echo "GNSS status log: $GNSS_STATUS"
echo "Trajectory bag: $TRAJ_BAG"
echo ""
