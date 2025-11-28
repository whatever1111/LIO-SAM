#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <fixposition_driver_msgs/FpaOdometry.h>
#include <std_msgs/Bool.h>
#include <fstream>
#include <iomanip>

class GNSSStatusMonitor
{
private:
    ros::NodeHandle nh;
    ros::Subscriber fpa_odom_sub;
    ros::Publisher gnss_degraded_pub;

    std::ofstream status_file;
    std::string output_file;

    bool is_degraded;
    double last_degraded_time;
    double degraded_start_time;

public:
    GNSSStatusMonitor() : nh("~"), is_degraded(false), last_degraded_time(0), degraded_start_time(-1)
    {
        nh.param<std::string>("output_file", output_file, "/tmp/gnss_status.txt");

        fpa_odom_sub = nh.subscribe("/fixposition/fpa/odometry", 10, &GNSSStatusMonitor::fpaOdomCallback, this);
        gnss_degraded_pub = nh.advertise<std_msgs::Bool>("/gnss_degraded", 10);

        status_file.open(output_file);
        status_file << "timestamp,gnss1_status,gnss2_status,is_degraded,degraded_duration\n";

        ROS_INFO("GNSS Status Monitor started, output: %s", output_file.c_str());
    }

    ~GNSSStatusMonitor()
    {
        if (status_file.is_open())
            status_file.close();
    }

    void fpaOdomCallback(const fixposition_driver_msgs::FpaOdometry::ConstPtr& msg)
    {
        int gnss1_status = msg->gnss1_status;
        int gnss2_status = msg->gnss2_status;
        double timestamp = msg->header.stamp.toSec();

        // Check if degraded (status == 5 means not fix)
        bool current_degraded = (gnss1_status == 5 || gnss2_status == 5);

        // Publish degraded status
        std_msgs::Bool degraded_msg;
        degraded_msg.data = current_degraded;
        gnss_degraded_pub.publish(degraded_msg);

        // Track degradation periods
        if (current_degraded && !is_degraded)
        {
            // Start of degradation
            degraded_start_time = timestamp;
            ROS_WARN("GNSS degradation detected at t=%.3f (GNSS1=%d, GNSS2=%d)",
                     timestamp, gnss1_status, gnss2_status);
        }
        else if (!current_degraded && is_degraded)
        {
            // End of degradation
            double duration = timestamp - degraded_start_time;
            ROS_INFO("GNSS recovered at t=%.3f, degradation duration: %.3f s",
                     timestamp, duration);
        }

        // Calculate current degradation duration
        double degraded_duration = 0.0;
        if (current_degraded)
        {
            if (degraded_start_time > 0)
                degraded_duration = timestamp - degraded_start_time;
        }

        // Log to file
        status_file << std::fixed << std::setprecision(6)
                   << timestamp << ","
                   << gnss1_status << ","
                   << gnss2_status << ","
                   << current_degraded << ","
                   << degraded_duration << "\n";

        is_degraded = current_degraded;
        last_degraded_time = timestamp;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gnss_status_monitor");
    GNSSStatusMonitor monitor;
    ros::spin();
    return 0;
}
