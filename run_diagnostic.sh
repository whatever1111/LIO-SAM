#!/bin/bash

echo "========================================"
echo "LIO-SAM Diagnostic System"
echo "========================================"
echo ""

rosparam set /use_sim_time true 

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  monitor    - Run real-time monitoring (with your evaluation script)"
    echo "  analyze    - Analyze existing CSV diagnostic data"
    echo "  visualize  - Visualize diagnostic data in real-time"
    echo ""
    echo "Examples:"
    echo "  $0 monitor                                          # Run monitoring with evaluation"
    echo "  $0 analyze /tmp/lio_sam_diagnostic_xxx.csv         # Analyze CSV file"
    echo "  $0 visualize /tmp/lio_sam_diagnostic_xxx.csv       # Visualize CSV file"
    echo ""
    exit 1
fi

MODE=$1
CATKIN_WS="/root/autodl-tmp/catkin_ws"
SCRIPT_DIR="$CATKIN_WS/src/LIO-SAM/scripts"

# Source ROS environment
source "$CATKIN_WS/devel/setup.bash"

case $MODE in
    monitor)
        echo "Starting diagnostic monitoring system..."
        echo "========================================"
        echo ""
        echo "This will:"
        echo "1. Start the diagnostic monitor"
        echo "2. Run your evaluation script"
        echo "3. Monitor all topics for anomalies"
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        echo ""

        # 启动诊断监控器
        echo "Starting diagnostic monitor..."
        python3 "$SCRIPT_DIR/diagnostic_monitor.py" &
        MONITOR_PID=$!

        sleep 2

        # 运行评估脚本
        echo ""
        echo "Starting LIO-SAM evaluation..."
        bash "$CATKIN_WS/src/LIO-SAM/run_evaluation.sh"

        # 等待用户中断
        echo ""
        echo "Evaluation completed. Monitor is still running."
        echo "Press Ctrl+C to stop the monitor and see the analysis."

        wait $MONITOR_PID

        # 找到最新的CSV文件
        LATEST_CSV=$(ls -t /tmp/lio_sam_diagnostic_*.csv 2>/dev/null | head -1)

        if [ -n "$LATEST_CSV" ]; then
            echo ""
            echo "Analyzing diagnostic data..."
            python3 "$SCRIPT_DIR/analyze_diagnostic.py" "$LATEST_CSV"
        fi
        ;;

    analyze)
        if [ "$#" -lt 2 ]; then
            echo "Error: Please provide CSV file path"
            echo "Usage: $0 analyze <csv_file>"
            exit 1
        fi

        CSV_FILE=$2

        if [ ! -f "$CSV_FILE" ]; then
            echo "Error: File not found: $CSV_FILE"
            exit 1
        fi

        echo "Analyzing diagnostic data from: $CSV_FILE"
        python3 "$SCRIPT_DIR/analyze_diagnostic.py" "$CSV_FILE"
        ;;

    visualize)
        if [ "$#" -lt 2 ]; then
            echo "Error: Please provide CSV file path"
            echo "Usage: $0 visualize <csv_file>"
            exit 1
        fi

        CSV_FILE=$2

        if [ ! -f "$CSV_FILE" ]; then
            echo "Error: File not found: $CSV_FILE"
            exit 1
        fi

        echo "Starting visualization of: $CSV_FILE"
        echo "Note: This will show real-time updates as new data is written to the CSV"
        python3 "$SCRIPT_DIR/visualize_diagnostic.py" "$CSV_FILE"
        ;;

    *)
        echo "Error: Unknown mode: $MODE"
        echo "Valid modes are: monitor, analyze, visualize"
        exit 1
        ;;
esac

echo ""
echo "Diagnostic system finished."