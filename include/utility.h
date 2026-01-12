#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_
#define PCL_NO_PRECOMPILE 

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <fixposition_driver_msgs/FpaImu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, LIVOX };

class ParamServer
{
public:

    ros::NodeHandle nh;

    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;
    bool useFpaImu;  // Use fixposition_driver_msgs/FpaImu instead of sensor_msgs/Imu

    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // GPS Weight Control
    float gpsNoiseMin;        // Minimum GPS noise (m), smaller = higher weight
    float gpsNoiseScale;      // Scale factor for GPS noise covariance
    float gpsAddInterval;     // Add GPS factor every N meters
    double gpsTimeWindow;     // Associate GPS to keyframe if |dt| <= window (seconds)
    string gpsTimeSyncMode;   // GPS association policy within the window: latest_before|nearest|first
    float gpsInitWaitDist;    // Wait for initial travel distance before enabling GPS factors (meters)
    float gpsPosStdFloor;     // Minimum GPS position std dev (m) for covariance floor
    float gpsOriStdFloorDeg;  // Minimum GPS orientation std dev (deg) for covariance floor
    string gpsRobustKernel;   // Robust kernel for GPS factors: none|huber|cauchy|tukey
    double gpsRobustDelta;    // Robust kernel delta (in whitened error units)

    // GPS Covariance Settings
    bool useGpsSensorCovariance;    // Use sensor-provided covariance instead of fixed values
    bool useGpsOrientationCov;      // Use GPS orientation covariance for heading constraint
    bool gpsYawOnly;               // If true and useGpsOrientationCov=true, only constrain yaw (roll/pitch weak)

    // GNSS-quality-aware GPS weighting (optional)
    bool useGnssDegraded;
    string gnssDegradedTopic;
    float gpsNoiseScaleGood;
    float gpsNoiseScaleDegraded;
    float gpsAddIntervalGood;
    float gpsAddIntervalDegraded;
    bool disablePoseCovGateWhenGnssGood;
    bool skipGpsWhenGnssDegraded;

    // GPS Extrinsics (ENU to LiDAR frame)
    vector<double> gpsExtRotV;
    Eigen::Matrix3d gpsExtRot;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;
    double lidarTimeOffset;  // Time offset to align LiDAR with IMU (seconds)

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    // IMU correction factor noise (LiDAR pose prior in imuPreintegration)
    // Order: [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z] in [rad, rad, rad, m, m, m]
    vector<double> imuCorrectionNoise;
    vector<double> imuCorrectionNoiseDegenerate;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;
    float degenerateEigenThreshold;  // eigenvalue threshold for scan-to-map degeneracy detection

    // Scan-to-map optimization
    int scan2MapMaxIterations;          // max LM iterations per scan
    float scan2MapConvergeDeltaRDeg;    // convergence threshold for rotation update (deg)
    float scan2MapConvergeDeltaTCm;     // convergence threshold for translation update (cm)

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer()
    {
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>("lio_sam/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("lio_sam/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("lio_sam/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");
        nh.param<bool>("lio_sam/useFpaImu", useFpaImu, false);  // Default to standard sensor_msgs/Imu

        nh.param<std::string>("lio_sam/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("lio_sam/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("lio_sam/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("lio_sam/mapFrame", mapFrame, "map");

        nh.param<bool>("lio_sam/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("lio_sam/useGpsElevation", useGpsElevation, false);
        nh.param<float>("lio_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("lio_sam/poseCovThreshold", poseCovThreshold, 25.0);

        // GPS Weight Control parameters
        nh.param<float>("lio_sam/gpsNoiseMin", gpsNoiseMin, 1.0);
        nh.param<float>("lio_sam/gpsNoiseScale", gpsNoiseScale, 1.0);
        nh.param<float>("lio_sam/gpsAddInterval", gpsAddInterval, 5.0);
        nh.param<double>("lio_sam/gpsTimeWindow", gpsTimeWindow, 0.2);
        nh.param<std::string>("lio_sam/gpsTimeSyncMode", gpsTimeSyncMode, std::string("latest_before"));
        nh.param<float>("lio_sam/gpsInitWaitDist", gpsInitWaitDist, 3.0);
        nh.param<float>("lio_sam/gpsPosStdFloor", gpsPosStdFloor, 0.2);
        nh.param<float>("lio_sam/gpsOriStdFloorDeg", gpsOriStdFloorDeg, 1.0);
        nh.param<std::string>("lio_sam/gpsRobustKernel", gpsRobustKernel, "huber");
        nh.param<double>("lio_sam/gpsRobustDelta", gpsRobustDelta, 1.345);

        // GPS Covariance Settings
        nh.param<bool>("lio_sam/useGpsSensorCovariance", useGpsSensorCovariance, true);
        nh.param<bool>("lio_sam/useGpsOrientationCov", useGpsOrientationCov, false);
        nh.param<bool>("lio_sam/gpsYawOnly", gpsYawOnly, false);

        // GNSS-quality-aware GPS weighting (optional)
        nh.param<bool>("lio_sam/useGnssDegraded", useGnssDegraded, false);
        nh.param<std::string>("lio_sam/gnssDegradedTopic", gnssDegradedTopic, "/gnss_degraded");
        nh.param<float>("lio_sam/gpsNoiseScaleGood", gpsNoiseScaleGood, gpsNoiseScale);
        nh.param<float>("lio_sam/gpsNoiseScaleDegraded", gpsNoiseScaleDegraded, gpsNoiseScale);
        nh.param<float>("lio_sam/gpsAddIntervalGood", gpsAddIntervalGood, gpsAddInterval);
        nh.param<float>("lio_sam/gpsAddIntervalDegraded", gpsAddIntervalDegraded, gpsAddInterval);
        nh.param<bool>("lio_sam/disablePoseCovGateWhenGnssGood", disablePoseCovGateWhenGnssGood, false);
        nh.param<bool>("lio_sam/skipGpsWhenGnssDegraded", skipGpsWhenGnssDegraded, false);

        // GPS extrinsics (ENU to LiDAR frame rotation)
        // Default is identity matrix (no rotation)
        nh.param<vector<double>>("lio_sam/gpsExtrinsicRot", gpsExtRotV, vector<double>());
        if (gpsExtRotV.size() == 9) {
            gpsExtRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(gpsExtRotV.data(), 3, 3);
        } else {
            gpsExtRot = Eigen::Matrix3d::Identity();
            if (!gpsExtRotV.empty()) {
                ROS_WARN("gpsExtrinsicRot should have 9 elements, using identity matrix");
            }
        }

        nh.param<bool>("lio_sam/savePCD", savePCD, false);
        nh.param<std::string>("lio_sam/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        std::string sensorStr;
        nh.param<std::string>("lio_sam/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster' or 'livox'): " << sensorStr);
            ros::shutdown();
        }

        nh.param<int>("lio_sam/N_SCAN", N_SCAN, 16);
        nh.param<int>("lio_sam/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>("lio_sam/downsampleRate", downsampleRate, 1);
        nh.param<float>("lio_sam/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("lio_sam/lidarMaxRange", lidarMaxRange, 1000.0);
        nh.param<double>("lio_sam/lidarTimeOffset", lidarTimeOffset, 0.0);

        nh.param<float>("lio_sam/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("lio_sam/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("lio_sam/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("lio_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);
        nh.param<float>("lio_sam/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<vector<double>>("lio_sam/imuCorrectionNoise", imuCorrectionNoise, vector<double>());
        nh.param<vector<double>>("lio_sam/imuCorrectionNoiseDegenerate", imuCorrectionNoiseDegenerate, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("lio_sam/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY).inverse();

        nh.param<float>("lio_sam/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("lio_sam/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("lio_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("lio_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);
        nh.param<float>("lio_sam/degenerateEigenThreshold", degenerateEigenThreshold, 10.0);

        nh.param<int>("lio_sam/scan2MapMaxIterations", scan2MapMaxIterations, 30);
        nh.param<float>("lio_sam/scan2MapConvergeDeltaRDeg", scan2MapConvergeDeltaRDeg, 0.05);
        nh.param<float>("lio_sam/scan2MapConvergeDeltaTCm", scan2MapConvergeDeltaTCm, 0.05);

        nh.param<float>("lio_sam/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("lio_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("lio_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("lio_sam/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("lio_sam/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("lio_sam/numberOfCores", numberOfCores, 2);
        nh.param<double>("lio_sam/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>("lio_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("lio_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("lio_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("lio_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("lio_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("lio_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("lio_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("lio_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("lio_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("lio_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("lio_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("lio_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("lio_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("lio_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }

	    // Overload for FpaImu message - extracts sensor_msgs/Imu and applies coordinate transform
	    // Note: CORRIMU does not have orientation data (all zeros), only acc and gyro
	    // FPA IMU has gravity in +Z, GTSAM MakeSharedU expects gravity in +Z, so no Z-flip needed
	    // extRot (Rz(90°)) only rotates X-Y plane for heading alignment
	    sensor_msgs::Imu imuConverter(const fixposition_driver_msgs::FpaImu& fpa_imu_in)
	    {
	        sensor_msgs::Imu imu_in = fpa_imu_in.data;
	        sensor_msgs::Imu imu_out = imu_in;

        // rotate acceleration (extRot aligns IMU X-Y to LiDAR X-Y)
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();

        // rotate gyroscope (same transform as acceleration)
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

	        // CORRIMU has no orientation - set identity quaternion
	        imu_out.orientation.x = 0;
	        imu_out.orientation.y = 0;
	        imu_out.orientation.z = 0;
	        imu_out.orientation.w = 1;

	        return imu_out;
	    }
};

template<typename T>
sensor_msgs::PointCloud2 publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub.getNumSubscribers() != 0)
        thisPub.publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif
