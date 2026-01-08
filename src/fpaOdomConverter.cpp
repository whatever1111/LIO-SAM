#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <fixposition_driver_msgs/FpaOdometry.h>
#include <fixposition_driver_msgs/FpaOdomenu.h>
#include <Eigen/Dense>
#include <cmath>

class FpaOdomConverter
{
private:
    ros::NodeHandle nh;
    ros::Subscriber fpa_odometry_sub;
    ros::Subscriber fpa_odomenu_sub;
    ros::Publisher nav_odom_pub;

    // Fixposition provides multiple odometry message types:
    //  - FpaOdometry: typically in ECEF (requires conversion to local ENU for LIO-SAM)
    //  - FpaOdomenu: already in local ENU (can be passed through)
    std::string input_type;  // "odometry"(ECEF) or "odomenu"(ENU)
    std::string input_topic;
    std::string output_topic;
    std::string output_frame;
    // If non-empty, override nav_msgs/Odometry.child_frame_id (pose frame name).
    // Useful to avoid treating FP_POI as a static mounting frame name in the robot TF tree.
    std::string pose_frame_override;
    // If true, override the message header stamp with ros::Time::now().
    // This can be useful when replaying bags where header stamps and bag time are inconsistent.
    bool use_receive_time;
    // If true, subtract the first received position as origin (useful for ENU odomenu streams).
    bool zero_initial_position;

    // ECEF to local ENU conversion
    bool origin_initialized;
    Eigen::Vector3d origin_ecef;
    Eigen::Matrix3d R_ecef_to_enu;
    bool enu_origin_initialized;
    Eigen::Vector3d origin_enu;

public:
    FpaOdomConverter() : nh("~"), origin_initialized(false), enu_origin_initialized(false)
    {
        // Get parameters
        nh.param<std::string>("input_type", input_type, "odometry");
        nh.param<std::string>("input_topic", input_topic, "/fixposition/fpa/odometry");
        nh.param<std::string>("output_topic", output_topic, "/odometry/gps");
        nh.param<std::string>("output_frame", output_frame, "odom");
        nh.param<std::string>("pose_frame_override", pose_frame_override, "");
        nh.param<bool>("use_receive_time", use_receive_time, false);
        nh.param<bool>("zero_initial_position", zero_initial_position, true);

        // Setup subscriber and publisher
        if (input_type == "odomenu") {
            fpa_odomenu_sub = nh.subscribe(input_topic, 200, &FpaOdomConverter::fpaOdomenuCallback, this);
        } else if (input_type == "odometry") {
            fpa_odometry_sub = nh.subscribe(input_topic, 200, &FpaOdomConverter::fpaOdometryCallback, this);
        } else {
            ROS_ERROR("Invalid input_type: %s (must be 'odometry' or 'odomenu')", input_type.c_str());
            ros::shutdown();
            return;
        }
        nav_odom_pub = nh.advertise<nav_msgs::Odometry>(output_topic, 200);

        ROS_INFO("FPA Odometry Converter started");
        ROS_INFO("Input type: %s", input_type.c_str());
        ROS_INFO("Input topic: %s", input_topic.c_str());
        ROS_INFO("Output topic: %s", output_topic.c_str());
        if (!pose_frame_override.empty())
            ROS_INFO("pose_frame_override: %s", pose_frame_override.c_str());
        ROS_INFO("use_receive_time: %s", use_receive_time ? "true" : "false");
        ROS_INFO("zero_initial_position: %s", zero_initial_position ? "true" : "false");
        if (input_type == "odometry") {
            ROS_INFO("Mode: ECEF -> local ENU (with covariance rotation)");
        } else {
            ROS_INFO("Mode: ENU passthrough (no ECEF conversion)");
        }
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

    // Convert ROS 6x6 covariance array (row-major) to Eigen matrix.
    static Eigen::Matrix<double, 6, 6> arrayToCov6(const boost::array<double, 36>& cov_in)
    {
        Eigen::Matrix<double, 6, 6> cov;
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                cov(i, j) = cov_in[i * 6 + j];
            }
        }
        return cov;
    }

    // Convert Eigen 6x6 covariance matrix to ROS row-major array.
    static void cov6ToArray(const Eigen::Matrix<double, 6, 6>& cov_in, boost::array<double, 36>& cov_out)
    {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                cov_out[i * 6 + j] = cov_in(i, j);
            }
        }
    }

    // Rotate a 6x6 covariance with layout [pos(3), rot(3)] using block-diagonal rotation T=diag(R,R).
    // Assumes rotation block is a small-angle vector in the parent frame.
    static Eigen::Matrix<double, 6, 6> rotateCov6(const Eigen::Matrix<double, 6, 6>& cov_in, const Eigen::Matrix3d& R)
    {
        Eigen::Matrix<double, 6, 6> T = Eigen::Matrix<double, 6, 6>::Zero();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 3>(3, 3) = R;
        Eigen::Matrix<double, 6, 6> cov_out = T * cov_in * T.transpose();
        return 0.5 * (cov_out + cov_out.transpose());  // enforce symmetry
    }

    void publishOdom(const std_msgs::Header& header,
                     const std::string& pose_frame,
                     const geometry_msgs::PoseWithCovariance& pose,
                     const geometry_msgs::TwistWithCovariance& velocity,
                     bool input_is_ecef)
    {
        nav_msgs::Odometry nav_odom;
        nav_odom.header = header;
        if (use_receive_time) {
            nav_odom.header.stamp = ros::Time::now();
        }
        nav_odom.header.frame_id = output_frame;
        nav_odom.child_frame_id = pose_frame_override.empty() ? pose_frame : pose_frame_override;

        if (!input_is_ecef) {
            // ENU mode: pose/twist are already in a local navigation frame.
            nav_odom.pose = pose;
            nav_odom.twist = velocity;
            if (zero_initial_position) {
                if (!enu_origin_initialized) {
                    origin_enu = Eigen::Vector3d(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
                    enu_origin_initialized = true;
                }
                nav_odom.pose.pose.position.x -= origin_enu.x();
                nav_odom.pose.pose.position.y -= origin_enu.y();
                nav_odom.pose.pose.position.z -= origin_enu.z();
            }
            nav_odom_pub.publish(nav_odom);
            return;
        }

        // ECEF mode: convert pose/twist (and their covariances) into the local ENU navigation frame.
        // Translation does not affect covariance, but rotation does.
        Eigen::Vector3d ecef_pos(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
        Eigen::Vector3d enu_pos = ecefToEnu(ecef_pos);
        nav_odom.pose.pose.position.x = enu_pos.x();
        nav_odom.pose.pose.position.y = enu_pos.y();
        nav_odom.pose.pose.position.z = enu_pos.z();

        // Convert orientation (assumes pose orientation is expressed in ECEF frame).
        if (origin_initialized) {
            Eigen::Quaterniond q_ecef(pose.pose.orientation.w, pose.pose.orientation.x,
                                      pose.pose.orientation.y, pose.pose.orientation.z);
            Eigen::Quaterniond q_rot(R_ecef_to_enu);
            Eigen::Quaterniond q_enu = q_rot * q_ecef;
            q_enu.normalize();

            nav_odom.pose.pose.orientation.x = q_enu.x();
            nav_odom.pose.pose.orientation.y = q_enu.y();
            nav_odom.pose.pose.orientation.z = q_enu.z();
            nav_odom.pose.pose.orientation.w = q_enu.w();
        } else {
            nav_odom.pose.pose.orientation = pose.pose.orientation;
        }

        // Rotate full 6x6 pose covariance: ROS convention uses [pos, rot] where rot is a small-angle vector.
        Eigen::Matrix<double, 6, 6> poseCov_ecef = arrayToCov6(pose.covariance);
        Eigen::Matrix<double, 6, 6> poseCov_enu = origin_initialized ? rotateCov6(poseCov_ecef, R_ecef_to_enu) : poseCov_ecef;
        cov6ToArray(poseCov_enu, nav_odom.pose.covariance);

        // Twist (velocity) - rotate linear & angular vectors and covariance to ENU
        if (origin_initialized) {
            Eigen::Vector3d lin_ecef(velocity.twist.linear.x, velocity.twist.linear.y, velocity.twist.linear.z);
            Eigen::Vector3d ang_ecef(velocity.twist.angular.x, velocity.twist.angular.y, velocity.twist.angular.z);
            Eigen::Vector3d lin_enu = R_ecef_to_enu * lin_ecef;
            Eigen::Vector3d ang_enu = R_ecef_to_enu * ang_ecef;

            nav_odom.twist.twist.linear.x = lin_enu.x();
            nav_odom.twist.twist.linear.y = lin_enu.y();
            nav_odom.twist.twist.linear.z = lin_enu.z();
            nav_odom.twist.twist.angular.x = ang_enu.x();
            nav_odom.twist.twist.angular.y = ang_enu.y();
            nav_odom.twist.twist.angular.z = ang_enu.z();

            Eigen::Matrix<double, 6, 6> twistCov_ecef = arrayToCov6(velocity.covariance);
            Eigen::Matrix<double, 6, 6> twistCov_enu = rotateCov6(twistCov_ecef, R_ecef_to_enu);
            cov6ToArray(twistCov_enu, nav_odom.twist.covariance);
        } else {
            nav_odom.twist = velocity;
        }

        nav_odom_pub.publish(nav_odom);
    }

    void fpaOdometryCallback(const fixposition_driver_msgs::FpaOdometry::ConstPtr& fpa_msg)
    {
        publishOdom(fpa_msg->header, fpa_msg->pose_frame, fpa_msg->pose, fpa_msg->velocity, true);
    }

    void fpaOdomenuCallback(const fixposition_driver_msgs::FpaOdomenu::ConstPtr& fpa_msg)
    {
        publishOdom(fpa_msg->header, fpa_msg->pose_frame, fpa_msg->pose, fpa_msg->velocity, false);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "fpa_odom_converter");

    FpaOdomConverter converter;

    ros::spin();

    return 0;
}
