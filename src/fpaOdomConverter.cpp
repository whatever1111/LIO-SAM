#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <fixposition_driver_msgs/FpaOdometry.h>
#include <Eigen/Dense>
#include <cmath>

class FpaOdomConverter
{
private:
    ros::NodeHandle nh;
    ros::Subscriber fpa_odom_sub;
    ros::Publisher nav_odom_pub;

    std::string input_topic;
    std::string output_topic;
    std::string output_frame;

    // ECEF to local ENU conversion
    bool origin_initialized;
    Eigen::Vector3d origin_ecef;
    Eigen::Matrix3d R_ecef_to_enu;

public:
    FpaOdomConverter() : nh("~"), origin_initialized(false)
    {
        // Get parameters
        nh.param<std::string>("input_topic", input_topic, "/fixposition/fpa/odometry");
        nh.param<std::string>("output_topic", output_topic, "/odometry/gps");
        nh.param<std::string>("output_frame", output_frame, "odom");

        // Setup subscriber and publisher
        fpa_odom_sub = nh.subscribe(input_topic, 200, &FpaOdomConverter::fpaOdomCallback, this);
        nav_odom_pub = nh.advertise<nav_msgs::Odometry>(output_topic, 200);

        ROS_INFO("FPA Odometry Converter started");
        ROS_INFO("Input topic: %s", input_topic.c_str());
        ROS_INFO("Output topic: %s", output_topic.c_str());
        ROS_INFO("Will convert ECEF coordinates to local ENU frame");
    }

    void initializeOrigin(const Eigen::Vector3d& ecef_pos)
    {
        origin_ecef = ecef_pos;

        // Convert ECEF to LLA to get latitude/longitude for rotation matrix
        double x = ecef_pos.x();
        double y = ecef_pos.y();
        double z = ecef_pos.z();

        // WGS84 parameters
        const double a = 6378137.0;  // semi-major axis
        const double f = 1.0 / 298.257223563;  // flattening
        const double e2 = 2.0 * f - f * f;  // first eccentricity squared

        // Calculate latitude and longitude
        double lon = atan2(y, x);
        double p = sqrt(x*x + y*y);
        double lat = atan2(z, p * (1.0 - e2));

        // Iterative refinement for latitude
        for (int i = 0; i < 5; i++) {
            double N = a / sqrt(1.0 - e2 * sin(lat) * sin(lat));
            double h = p / cos(lat) - N;
            lat = atan2(z, p * (1.0 - e2 * N / (N + h)));
        }

        // Build rotation matrix from ECEF to ENU
        double sin_lat = sin(lat);
        double cos_lat = cos(lat);
        double sin_lon = sin(lon);
        double cos_lon = cos(lon);

        R_ecef_to_enu << -sin_lon,           cos_lon,          0,
                         -sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat,
                          cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat;

        origin_initialized = true;

        ROS_INFO("Origin initialized at ECEF: [%.2f, %.2f, %.2f]",
                 origin_ecef.x(), origin_ecef.y(), origin_ecef.z());
        ROS_INFO("Latitude: %.6f deg, Longitude: %.6f deg",
                 lat * 180.0 / M_PI, lon * 180.0 / M_PI);
    }

    Eigen::Vector3d ecefToEnu(const Eigen::Vector3d& ecef_pos)
    {
        if (!origin_initialized) {
            initializeOrigin(ecef_pos);
            return Eigen::Vector3d::Zero();  // First position is origin
        }

        // Convert to local ENU
        Eigen::Vector3d delta_ecef = ecef_pos - origin_ecef;
        Eigen::Vector3d enu = R_ecef_to_enu * delta_ecef;

        // ENU: East(+X), North(+Y), Up(+Z)
        // LIO-SAM now uses +X forward (after Livox rotation in code)
        // So we return ENU directly without additional rotation
        return enu;
    }

    void fpaOdomCallback(const fixposition_driver_msgs::FpaOdometry::ConstPtr& fpa_msg)
    {
        // Convert FpaOdometry to nav_msgs::Odometry
        nav_msgs::Odometry nav_odom;

        // Header
        nav_odom.header = fpa_msg->header;
        nav_odom.header.frame_id = output_frame;

        // Child frame
        nav_odom.child_frame_id = fpa_msg->pose_frame;

        // Convert ECEF position to local ENU
        Eigen::Vector3d ecef_pos(
            fpa_msg->pose.pose.position.x,
            fpa_msg->pose.pose.position.y,
            fpa_msg->pose.pose.position.z
        );

        Eigen::Vector3d enu_pos = ecefToEnu(ecef_pos);

        // Set converted position
        nav_odom.pose.pose.position.x = enu_pos.x();
        nav_odom.pose.pose.position.y = enu_pos.y();
        nav_odom.pose.pose.position.z = enu_pos.z();

        // Convert orientation from ECEF to ENU frame
        // FPA quaternion is in ECEF frame, need to rotate to ENU
        if (origin_initialized) {
            Eigen::Quaterniond q_ecef(
                fpa_msg->pose.pose.orientation.w,
                fpa_msg->pose.pose.orientation.x,
                fpa_msg->pose.pose.orientation.y,
                fpa_msg->pose.pose.orientation.z
            );

            // R_ecef_to_enu transforms vectors from ECEF to ENU
            // For quaternion: q_enu = R_ecef_to_enu * q_ecef
            Eigen::Quaterniond q_rot(R_ecef_to_enu);
            Eigen::Quaterniond q_enu = q_rot * q_ecef;

            nav_odom.pose.pose.orientation.x = q_enu.x();
            nav_odom.pose.pose.orientation.y = q_enu.y();
            nav_odom.pose.pose.orientation.z = q_enu.z();
            nav_odom.pose.pose.orientation.w = q_enu.w();
        } else {
            nav_odom.pose.pose.orientation = fpa_msg->pose.pose.orientation;
        }

        // Copy covariance
        nav_odom.pose.covariance = fpa_msg->pose.covariance;

        // Twist (velocity) - rotate from ECEF to ENU
        if (origin_initialized) {
            Eigen::Vector3d ecef_vel(
                fpa_msg->velocity.twist.linear.x,
                fpa_msg->velocity.twist.linear.y,
                fpa_msg->velocity.twist.linear.z
            );
            Eigen::Vector3d enu_vel = R_ecef_to_enu * ecef_vel;

            // ENU velocity directly (no additional rotation needed)
            nav_odom.twist.twist.linear.x = enu_vel.x();
            nav_odom.twist.twist.linear.y = enu_vel.y();
            nav_odom.twist.twist.linear.z = enu_vel.z();
        } else {
            nav_odom.twist.twist.linear = fpa_msg->velocity.twist.linear;
        }

        // Angular velocity stays the same
        nav_odom.twist.twist.angular = fpa_msg->velocity.twist.angular;
        nav_odom.twist.covariance = fpa_msg->velocity.covariance;

        // Publish
        nav_odom_pub.publish(nav_odom);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fpa_odom_converter");

    FpaOdomConverter converter;

    ros::spin();

    return 0;
}
