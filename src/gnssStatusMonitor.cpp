#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <fixposition_driver_msgs/FpaOdometry.h>
#include <fixposition_driver_msgs/FpaOdomenu.h>
#include <std_msgs/Bool.h>
#include <fstream>
#include <iomanip>

class GNSSStatusMonitor
{
private:
    ros::NodeHandle nh;
    ros::Subscriber fpa_odom_sub;
    ros::Subscriber fpa_odomenu_sub;
    ros::Publisher gnss_degraded_pub;

    std::ofstream status_file;
    std::string output_file;
    // Fixposition publishes GNSS quality as integer fix types (see fixposition_driver_msgs/FpaConsts.msg).
    // We convert it into a simple boolean "/gnss_degraded" signal for gating GPS factors.
    std::string input_type;
    std::string input_topic;
    bool use_receive_time;  // Use ros::Time::now() as time base for logging (bag-time friendly)
    int min_fix_for_good;   // Threshold at/above which GNSS is considered "good" (e.g. 7=float, 8=fixed)
    bool degraded_if_any_below_min;  // true: degraded if either antenna below min; false: require both below min

    bool is_degraded;
    double last_degraded_time;
    double degraded_start_time;

public:
    GNSSStatusMonitor() : nh("~"), is_degraded(false), last_degraded_time(0), degraded_start_time(-1)
    {
        nh.param<std::string>("output_file", output_file, "/tmp/gnss_status.txt");
        nh.param<std::string>("input_type", input_type, "odometry");  // "odometry" or "odomenu"
        nh.param<std::string>("input_topic", input_topic, "/fixposition/fpa/odometry");
        nh.param<bool>("use_receive_time", use_receive_time, true);
        nh.param<int>("min_fix_for_good", min_fix_for_good, 7);  // 7=RTK_FLOAT, 8=RTK_FIXED
        nh.param<bool>("degraded_if_any_below_min", degraded_if_any_below_min, true);

        if (input_type == "odomenu") {
            fpa_odomenu_sub = nh.subscribe(input_topic, 200, &GNSSStatusMonitor::fpaOdomenuCallback, this);
        } else if (input_type == "odometry") {
            fpa_odom_sub = nh.subscribe(input_topic, 200, &GNSSStatusMonitor::fpaOdomCallback, this);
        } else {
            ROS_ERROR("Invalid input_type: %s (must be 'odometry' or 'odomenu')", input_type.c_str());
            ros::shutdown();
            return;
        }
        gnss_degraded_pub = nh.advertise<std_msgs::Bool>("/gnss_degraded", 10);

        status_file.open(output_file);
        status_file << "ros_time,header_stamp,gnss1_status,gnss2_status,min_fix_for_good,is_degraded,degraded_duration\n";

        ROS_INFO("GNSS Status Monitor started, output: %s", output_file.c_str());
        ROS_INFO("Input type: %s, input topic: %s", input_type.c_str(), input_topic.c_str());
        ROS_INFO("use_receive_time: %s, min_fix_for_good: %d, degraded_if_any_below_min: %s",
                 use_receive_time ? "true" : "false", min_fix_for_good, degraded_if_any_below_min ? "true" : "false");
    }

    ~GNSSStatusMonitor()
    {
        if (status_file.is_open())
            status_file.close();
    }

    void fpaOdomCallback(const fixposition_driver_msgs::FpaOdometry::ConstPtr& msg)
    {
        handleStatuses(msg->gnss1_status, msg->gnss2_status, msg->header.stamp);
    }

    void fpaOdomenuCallback(const fixposition_driver_msgs::FpaOdomenu::ConstPtr& msg)
    {
        handleStatuses(msg->gnss1_status, msg->gnss2_status, msg->header.stamp);
    }

private:
    void handleStatuses(int gnss1_status, int gnss2_status, const ros::Time& header_stamp)
    {
        // NOTE: Fixposition FPA uses consts.GNSS_FIX_* for gnss*_status (see fixposition_driver_msgs/FpaConsts.msg)
        // e.g. 5=S3D, 7=RTK_FLOAT, 8=RTK_FIXED.
        double ros_time = use_receive_time ? ros::Time::now().toSec() : header_stamp.toSec();

        bool current_degraded = false;
        if (degraded_if_any_below_min) {
            current_degraded = (gnss1_status < min_fix_for_good) || (gnss2_status < min_fix_for_good);
        } else {
            current_degraded = (gnss1_status < min_fix_for_good) && (gnss2_status < min_fix_for_good);
        }

        // Publish degraded status (use ROS time base via bag recording time)
        std_msgs::Bool degraded_msg;
        degraded_msg.data = current_degraded;
        gnss_degraded_pub.publish(degraded_msg);

        // Track degradation periods
        if (current_degraded && !is_degraded)
        {
            degraded_start_time = ros_time;
            ROS_WARN("GNSS degraded at t=%.3f (GNSS1=%d, GNSS2=%d, min_fix_for_good=%d)",
                     ros_time, gnss1_status, gnss2_status, min_fix_for_good);
        }
        else if (!current_degraded && is_degraded)
        {
            double duration = ros_time - degraded_start_time;
            ROS_INFO("GNSS recovered at t=%.3f, degraded duration: %.3f s",
                     ros_time, duration);
        }

        double degraded_duration = 0.0;
        if (current_degraded && degraded_start_time > 0)
            degraded_duration = ros_time - degraded_start_time;

        // Log to file
        status_file << std::fixed << std::setprecision(6)
                    << ros_time << ","
                    << header_stamp.toSec() << ","
                    << gnss1_status << ","
                    << gnss2_status << ","
                    << min_fix_for_good << ","
                    << current_degraded << ","
                    << degraded_duration << "\n";
        status_file.flush();

        is_degraded = current_degraded;
        last_degraded_time = ros_time;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gnss_status_monitor");
    GNSSStatusMonitor monitor;
    ros::spin();
    return 0;
}
