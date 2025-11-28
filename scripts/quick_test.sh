#!/bin/bash
# Quick test script to verify LIO-SAM timestamp fix for Livox
# This runs the bag for 30 seconds and checks for velocity anomalies

BAG_FILE="${1:-/root/autodl-tmp/info_fixed.bag}"
DURATION="${2:-30}"

echo "=============================================="
echo "LIO-SAM Quick Test (Livox Timestamp Fix)"
echo "=============================================="
echo "Bag file: $BAG_FILE"
echo "Test duration: ${DURATION}s"
echo ""

# Check if bag file exists
if [ ! -f "$BAG_FILE" ]; then
    echo "ERROR: Bag file not found: $BAG_FILE"
    exit 1
fi

# Source ROS environment
source /opt/ros/noetic/setup.bash
source /root/autodl-tmp/catkin_ws/devel/setup.bash

# Create temp directory for logs
LOG_DIR="/tmp/liosam_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"
echo ""

# Start roscore
roscore &
ROSCORE_PID=$!
sleep 2

# Start LIO-SAM nodes
echo "Starting LIO-SAM nodes..."
roslaunch lio_sam run_evaluation.launch &
LIOSAM_PID=$!
sleep 3

# Start velocity monitor in background
echo "Starting velocity monitor..."
rostopic echo /lio_sam/mapping/odometry -p --noarr > "$LOG_DIR/odom.csv" &
ODOM_PID=$!

# Play bag for specified duration
echo "Playing bag for ${DURATION} seconds..."
rosbag play "$BAG_FILE" --clock --duration=$DURATION 2>&1 | tee "$LOG_DIR/bag_play.log"

# Wait a bit for last messages
sleep 2

# Kill all processes
echo ""
echo "Stopping processes..."
kill $ODOM_PID 2>/dev/null
kill $LIOSAM_PID 2>/dev/null
kill $ROSCORE_PID 2>/dev/null

sleep 2

# Analyze results
echo ""
echo "=============================================="
echo "Test Results"
echo "=============================================="

if [ -f "$LOG_DIR/odom.csv" ]; then
    # Count lines (excluding header)
    ODOM_COUNT=$(tail -n +2 "$LOG_DIR/odom.csv" | wc -l)
    echo "Odometry messages received: $ODOM_COUNT"

    if [ "$ODOM_COUNT" -gt 10 ]; then
        # Extract velocity components and calculate speeds
        echo ""
        echo "Velocity analysis:"
        tail -n +2 "$LOG_DIR/odom.csv" | awk -F',' '{
            vx = $8; vy = $9; vz = $10;
            speed = sqrt(vx*vx + vy*vy + vz*vz);
            if (speed > max_speed) max_speed = speed;
            if (speed > 10) spike_count++;
            total_speed += speed;
            count++;
        } END {
            if (count > 0) {
                print "  Mean speed: " total_speed/count " m/s";
                print "  Max speed: " max_speed " m/s";
                print "  Velocity spikes (>10 m/s): " spike_count;
                if (max_speed > 50) {
                    print "  STATUS: FAIL - Velocity anomaly detected!";
                } else {
                    print "  STATUS: OK - Velocities within normal range";
                }
            } else {
                print "  No valid data";
            }
        }' 2>/dev/null
    else
        echo "  Not enough odometry data"
    fi
else
    echo "No odometry data collected"
fi

echo ""
echo "Logs saved to: $LOG_DIR"
echo "=============================================="
