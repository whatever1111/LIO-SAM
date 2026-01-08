#include <ros/ros.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <Eigen/Dense>

#include <cmath>
#include <string>
#include <vector>

namespace
{

Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d &R)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d Rn = U * V.transpose();
    if (Rn.determinant() < 0.0)
    {
        U.col(2) *= -1.0;
        Rn = U * V.transpose();
    }
    return Rn;
}

bool loadVec3(ros::NodeHandle &nh, const std::string &key, Eigen::Vector3d &out, const Eigen::Vector3d &def)
{
    std::vector<double> v;
    nh.param<std::vector<double>>(key, v, std::vector<double>());
    if (v.size() == 3)
    {
        out = Eigen::Vector3d(v[0], v[1], v[2]);
        return true;
    }
    if (!v.empty())
    {
        ROS_WARN("Param %s should have 3 elements, using default", key.c_str());
    }
    out = def;
    return false;
}

bool loadMat3RowMajor(ros::NodeHandle &nh, const std::string &key, Eigen::Matrix3d &out, const Eigen::Matrix3d &def)
{
    std::vector<double> v;
    nh.param<std::vector<double>>(key, v, std::vector<double>());
    if (v.size() == 9)
    {
        out = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(v.data());
        return true;
    }
    if (!v.empty())
    {
        ROS_WARN("Param %s should have 9 elements, using default", key.c_str());
    }
    out = def;
    return false;
}

geometry_msgs::Quaternion quatMsgFromMat(const Eigen::Matrix3d &R_in)
{
    Eigen::Matrix3d R = projectToSO3(R_in);
    Eigen::Quaterniond q(R);
    q.normalize();
    geometry_msgs::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
}

geometry_msgs::TransformStamped makeStampedTf(
    const std::string &parent,
    const std::string &child,
    const Eigen::Vector3d &t_parent_child,
    const Eigen::Matrix3d &R_parent_child,
    const ros::Time &stamp)
{
    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = parent;
    tf_msg.child_frame_id = child;
    tf_msg.transform.translation.x = t_parent_child.x();
    tf_msg.transform.translation.y = t_parent_child.y();
    tf_msg.transform.translation.z = t_parent_child.z();
    tf_msg.transform.rotation = quatMsgFromMat(R_parent_child);
    return tf_msg;
}

} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lio_sam_tf_publisher");
    ros::NodeHandle nh;

    std::string baseFrame;
    std::string lidarFrame;
    std::string imuFrame;
    std::string gpsFrame;
    nh.param<std::string>("lio_sam/baselinkFrame", baseFrame, "base_link");
    nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "lidar_link");
    nh.param<std::string>("lio_sam/imuFrame", imuFrame, "imu_link");
    nh.param<std::string>("lio_sam/gpsFrame", gpsFrame, "gps_link");

    bool publishBaseToLidar = true;
    bool publishLidarToImu = true;
    bool publishBaseToGps = true;
    nh.param<bool>("lio_sam/publishBaseToLidarTf", publishBaseToLidar, true);
    nh.param<bool>("lio_sam/publishLidarToImuTf", publishLidarToImu, true);
    nh.param<bool>("lio_sam/publishBaseToGpsTf", publishBaseToGps, true);

    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Z = Eigen::Vector3d::Zero();

    // base_link -> lidar_link
    Eigen::Vector3d t_b_l;
    Eigen::Matrix3d R_b_l;
    loadVec3(nh, "lio_sam/baseToLidarTrans", t_b_l, Z);
    loadMat3RowMajor(nh, "lio_sam/baseToLidarRot", R_b_l, I);

    // lidar_link -> imu_link (from LIO-SAM extrinsic params)
    Eigen::Vector3d t_l_i_raw;
    Eigen::Matrix3d R_l_i;
    loadVec3(nh, "lio_sam/extrinsicTrans", t_l_i_raw, Z);

    // Prefer extrinsicRPY for frame rotation; fallback to extrinsicRot if missing.
    bool hasRpy = loadMat3RowMajor(nh, "lio_sam/extrinsicRPY", R_l_i, I);
    if (!hasRpy)
    {
        loadMat3RowMajor(nh, "lio_sam/extrinsicRot", R_l_i, I);
    }

    bool transIsLidarToImu = true;
    nh.param<bool>("lio_sam/extrinsicTransIsLidarToImu", transIsLidarToImu, true);
    Eigen::Vector3d t_l_i = t_l_i_raw;
    if (!transIsLidarToImu)
    {
        // Given t_i_l (LiDAR origin expressed in IMU frame), convert to t_l_i (IMU origin in LiDAR frame):
        // t_l_i = - R_l_i * t_i_l
        t_l_i = -projectToSO3(R_l_i) * t_l_i_raw;
        ROS_WARN_ONCE("extrinsicTransIsLidarToImu=false: interpreting extrinsicTrans as t_imu_lidar and converting to t_lidar_imu for TF publishing");
    }

    // base_link -> gps_link (TODO: measured by user; identity by default)
    Eigen::Vector3d t_b_g;
    Eigen::Matrix3d R_b_g;
    loadVec3(nh, "lio_sam/baseToGpsTrans", t_b_g, Z);
    loadMat3RowMajor(nh, "lio_sam/baseToGpsRot", R_b_g, I);

    ROS_INFO("TF Publisher frames:");
    ROS_INFO("  baseFrame: %s", baseFrame.c_str());
    ROS_INFO("  lidarFrame: %s", lidarFrame.c_str());
    ROS_INFO("  imuFrame: %s", imuFrame.c_str());
    ROS_INFO("  gpsFrame: %s", gpsFrame.c_str());
    ROS_INFO("TF Publisher enabled:");
    ROS_INFO("  base->lidar: %s", publishBaseToLidar ? "true" : "false");
    ROS_INFO("  lidar->imu:  %s", publishLidarToImu ? "true" : "false");
    ROS_INFO("  base->gps:   %s", publishBaseToGps ? "true" : "false");
    ROS_INFO("TF publish mode: /tf_static (latched), one-shot");

    tf2_ros::StaticTransformBroadcaster br;
    std::vector<geometry_msgs::TransformStamped> tfs;
    ros::Time stamp = ros::Time::now();
    if (stamp.isZero())
    {
        // Under /use_sim_time, ros::Time::now() is 0 until /clock starts. We still publish once
        // immediately (so nodes that need the static TF at startup can proceed), then republish once
        // /clock becomes valid so visualization tools that treat /tf_static stamps as time-bound
        // (e.g. Foxglove) don't show the TF tree as "split".
        ROS_INFO("Time is zero; publishing /tf_static with stamp=0 now, and will republish once /clock is valid...");
    }
    if (publishBaseToLidar)
    {
        tfs.push_back(makeStampedTf(baseFrame, lidarFrame, t_b_l, R_b_l, stamp));
    }
    if (publishLidarToImu)
    {
        tfs.push_back(makeStampedTf(lidarFrame, imuFrame, t_l_i, R_l_i, stamp));
    }
    if (publishBaseToGps)
    {
        tfs.push_back(makeStampedTf(baseFrame, gpsFrame, t_b_g, R_b_g, stamp));
    }
    br.sendTransform(tfs);

    // If sim time isn't valid yet, use a wall timer to republish once /clock starts ticking.
    // (WallTimer runs even when /use_sim_time=true and time==0.)
    bool republished = false;
    ros::WallTimer republish_timer;
    if (stamp.isZero())
    {
        republish_timer = nh.createWallTimer(
            ros::WallDuration(0.1),
            [&](const ros::WallTimerEvent &)
            {
                if (republished)
                    return;
                const ros::Time now = ros::Time::now();
                if (now.isZero())
                    return;

                for (auto &tf_msg : tfs)
                    tf_msg.header.stamp = now;
                br.sendTransform(tfs);
                republished = true;
                republish_timer.stop();
                ROS_INFO("Republished /tf_static with stamp=%.3f", now.toSec());
            },
            /*oneshot=*/false,
            /*autostart=*/true);
    }
    ros::spin();

    return 0;
}
