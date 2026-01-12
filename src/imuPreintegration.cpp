#include "utility.h"
#include "lio_sam/MappingStatus.h"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <boost/bind/bind.hpp>

#include <memory>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    Eigen::Affine3f lidarOdomAffine;
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener;
    tf::StampedTransform lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::Odometry> imuOdomQueue;

    TransformFusion()
    {
        // Default to identity so we never publish uninitialized quaternions if TF lookup fails
        // (common under /use_sim_time when bag playback hasn't started yet).
        lidar2Baselink = tf::StampedTransform(
            tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0)),
            ros::Time(0),
            lidarFrame,
            baselinkFrame
        );

        if(lidarFrame != baselinkFrame)
        {
            try
            {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex)
            {
                ROS_WARN("TransformFusion: cannot lookup TF %s -> %s (%s). Using identity for lidar<->base.",
                         lidarFrame.c_str(), baselinkFrame.c_str(), ex.what());
            }
        }

        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }

    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = odomMsg->header.stamp.toSec();
    }

    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf
        static tf::TransformBroadcaster tfMap2Odom;
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg);
        // Prevent unbounded growth if LiDAR odometry is missing.
        static const size_t kMaxImuOdomQueueSize = 2000;
        while (imuOdomQueue.size() > kMaxImuOdomQueueSize)
            imuOdomQueue.pop_front();

        // If we haven't received any LiDAR odometry yet, still publish a best-effort TF chain so
        // visualization/tools don't report "missing transform" (e.g., lidar_link -> map).
        // This uses the raw IMU incremental odometry pose directly until lidarOdomAffine becomes available.
        if (lidarOdomTime == -1)
        {
            nav_msgs::Odometry imuOdometry = imuOdomQueue.back();
            pubImuOdometry.publish(imuOdometry);

            // publish tf (odom -> base_link)
            static tf::TransformBroadcaster tfOdom2BaseLink;
            tf::Transform tCur;
            tf::poseMsgToTF(imuOdometry.pose.pose, tCur);
            if (lidarFrame != baselinkFrame)
                tCur = tCur * lidar2Baselink;
            tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
            tfOdom2BaseLink.sendTransform(odom_2_baselink);

            return;
        }

        // get latest odometry (at current IMU stamp)
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry);

        // publish tf
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // Note: base_link <-> lidarFrame should be provided by external static TF (e.g. lio_sam_tfPublisher).

        // publish IMU path
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    ros::Subscriber subImu;
    ros::Subscriber subFpaImu;  // For FpaImu message type
    ros::Subscriber subImuOrientation;  // For getting orientation from /imu/data in FPA mode

    // Mapping publishes a separate status message (degenerate/non-degenerate) instead of encoding it in covariance.
    // Synchronize odometry_incremental with MappingStatus so the IMU graph uses the correct flag.
    message_filters::Subscriber<nav_msgs::Odometry> subOdometry;
    message_filters::Subscriber<lio_sam::MappingStatus> subMappingStatus;
    using OdomStatusSyncPolicy =
        message_filters::sync_policies::ApproximateTime<nav_msgs::Odometry, lio_sam::MappingStatus>;
    std::shared_ptr<message_filters::Synchronizer<OdomStatusSyncPolicy>> odomStatusSync;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuWithOrientation;  // Publish IMU with integrated orientation for FpaImu

    bool systemInitialized = false;

    // CORRIMU orientation integration (since CORRIMU has no orientation data)
    Eigen::Quaterniond corrImuOrientation_ = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond latestImuOrientation_ = Eigen::Quaterniond::Identity();  // From /imu/data
    bool hasImuOrientation_ = false;
    double lastCorrImuTime_ = -1;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    std::deque<sensor_msgs::Imu> imuQueOpt;
    std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    // Since imuConverter() already transforms IMU data from IMU frame to LiDAR frame,
    // the preintegration is performed in LiDAR frame directly.
    // Therefore, no additional frame transformation is needed.
    // T_bl: identity (IMU data already in LiDAR frame after imuConverter)
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3::identity(), gtsam::Point3(0, 0, 0));
    // T_lb: identity
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3::identity(), gtsam::Point3(0, 0, 0));

    IMUPreintegration()
    {
        // Subscribe to IMU based on configuration
        if (useFpaImu) {
            subFpaImu = nh.subscribe<fixposition_driver_msgs::FpaImu>(imuTopic, 2000, &IMUPreintegration::fpaImuHandler, this, ros::TransportHints().tcpNoDelay());
            // Also subscribe to /imu/data to get orientation (before system init)
            subImuOrientation = nh.subscribe<sensor_msgs::Imu>("/imu/data", 200, &IMUPreintegration::imuOrientationHandler, this, ros::TransportHints().tcpNoDelay());
            ROS_INFO("IMU Preintegration: Subscribing to FpaImu topic: %s, orientation from /imu/data", imuTopic.c_str());
        } else {
            subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &IMUPreintegration::imuHandler, this, ros::TransportHints().tcpNoDelay());
            ROS_INFO("IMU Preintegration: Subscribing to standard Imu topic: %s", imuTopic.c_str());
        }
        // ApproximateTime avoids strict timestamp equality requirements while keeping the pairing stable.
        subOdometry.subscribe(nh, "lio_sam/mapping/odometry_incremental", 50, ros::TransportHints().tcpNoDelay());
        subMappingStatus.subscribe(nh, "lio_sam/mapping/odometry_incremental_status", 50, ros::TransportHints().tcpNoDelay());
        odomStatusSync = std::make_shared<message_filters::Synchronizer<OdomStatusSyncPolicy>>(
            OdomStatusSyncPolicy(50), subOdometry, subMappingStatus);
        odomStatusSync->registerCallback(
            boost::bind(&IMUPreintegration::odometryHandler, this, boost::placeholders::_1, boost::placeholders::_2));

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);

        // Publish IMU with integrated orientation (useful for debugging and other nodes)
        if (useFpaImu) {
            pubImuWithOrientation = nh.advertise<sensor_msgs::Imu>("lio_sam/imu/data", 2000);
            ROS_INFO("Publishing IMU with orientation to: lio_sam/imu/data");
        }

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e0); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2); // 1e-2 ~ 1e-3 seems to be good
        // LiDAR pose correction noise (used as a PriorFactor in the IMU graph).
        // These are sigmas in the IMU graph tangent space: [rot, rot, rot, pos, pos, pos].
        gtsam::Vector6 correctionSigmas;
        correctionSigmas << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1;
        if (imuCorrectionNoise.size() == 6)
        {
            correctionSigmas << imuCorrectionNoise[0], imuCorrectionNoise[1], imuCorrectionNoise[2],
                                 imuCorrectionNoise[3], imuCorrectionNoise[4], imuCorrectionNoise[5];
        }
        else if (!imuCorrectionNoise.empty())
        {
            ROS_WARN("imuCorrectionNoise expects 6 elements, using defaults");
        }
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas(correctionSigmas);

        gtsam::Vector6 correctionSigmasDeg;
        correctionSigmasDeg << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
        if (imuCorrectionNoiseDegenerate.size() == 6)
        {
            correctionSigmasDeg << imuCorrectionNoiseDegenerate[0], imuCorrectionNoiseDegenerate[1], imuCorrectionNoiseDegenerate[2],
                                    imuCorrectionNoiseDegenerate[3], imuCorrectionNoiseDegenerate[4], imuCorrectionNoiseDegenerate[5];
        }
        else if (!imuCorrectionNoiseDegenerate.empty())
        {
            ROS_WARN("imuCorrectionNoiseDegenerate expects 6 elements, using defaults");
        }
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas(correctionSigmasDeg);
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
        // Reset CORRIMU orientation integration
        corrImuOrientation_ = Eigen::Quaterniond::Identity();
        lastCorrImuTime_ = -1;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg,
                         const lio_sam::MappingStatus::ConstPtr& statusMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        if (imuQueOpt.empty())
            return;

        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        // When mapping is degenerate (poor constraints), use a weaker correction noise (correctionNoise2).
        bool degenerate = statusMsg ? statusMsg->is_degenerate : false;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        if (systemInitialized == false)
        {
            resetOptimization();

            // pop old IMU message
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            // Initialize CORRIMU orientation with LiDAR pose orientation
            // Note: For CORRIMU without native orientation, the initial yaw from LiDAR
            // may be inaccurate (since it was initialized from integration starting at identity).
            // The GPS factor will correct this over time.
            if (useFpaImu) {
                gtsam::Quaternion q = lidarPose.rotation().toQuaternion();
                corrImuOrientation_ = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());

                // Extract RPY for logging
                double roll, pitch, yaw;
                tf::Matrix3x3(tf::Quaternion(q.x(), q.y(), q.z(), q.w())).getRPY(roll, pitch, yaw);
                ROS_INFO("CORRIMU orientation initialized from LiDAR pose: Roll=%.1f, Pitch=%.1f, Yaw=%.1f deg",
                         roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
            }

            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        if (key == 100)
        {
            // get updated noise before reset
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            resetOptimization();
            // add pose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }


        // 1. integrate imu data and optimize
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            else
                break;
        }
        // add imu factor to graph
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // add imu bias between factor
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // Reset the optimization preintegration object.
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        // Correct CORRIMU integrated orientation with optimized pose to prevent drift
        if (useFpaImu) {
            gtsam::Quaternion q = prevPose_.rotation().toQuaternion();
            corrImuOrientation_ = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());
        }

        // check optimization
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        double lastImuQT = -1;
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }
        // repropogate
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;
        doneFirstOpt = true;
    }

    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    // Handler for /imu/data to get orientation (used in FPA mode before system init)
    void imuOrientationHandler(const sensor_msgs::Imu::ConstPtr& imu_msg)
    {
        // Apply extrinsicRPY transformation to orientation
        Eigen::Quaterniond q_raw(imu_msg->orientation.w, imu_msg->orientation.x,
                                  imu_msg->orientation.y, imu_msg->orientation.z);
        // extQRPY is inverse of extrinsicRPY rotation
        latestImuOrientation_ = q_raw * extQRPY;
        hasImuOrientation_ = true;

        // Debug: print orientation transformation (every 200 messages)
        static int debugCount = 0;
        if (debugCount++ % 200 == 0)
        {
            // Convert to Euler for human-readable output
            tf::Quaternion q_raw_tf(q_raw.x(), q_raw.y(), q_raw.z(), q_raw.w());
            tf::Quaternion q_trans_tf(latestImuOrientation_.x(), latestImuOrientation_.y(),
                                       latestImuOrientation_.z(), latestImuOrientation_.w());
            double r_raw, p_raw, y_raw, r_trans, p_trans, y_trans;
            tf::Matrix3x3(q_raw_tf).getRPY(r_raw, p_raw, y_raw);
            tf::Matrix3x3(q_trans_tf).getRPY(r_trans, p_trans, y_trans);

            ROS_INFO("imuOrientationHandler: raw yaw=%.1f deg, transformed yaw=%.1f deg (expected: raw-90=%.1f)",
                     y_raw * 180.0 / M_PI, y_trans * 180.0 / M_PI, (y_raw * 180.0 / M_PI) - 90.0);
        }
    }

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }

    // Handler for FpaImu messages
    // Integrates angular velocity from CORRIMU to compute pose quaternion
    void fpaImuHandler(const fixposition_driver_msgs::FpaImu::ConstPtr& fpa_imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        double imuTime = fpa_imu_raw->data.header.stamp.toSec();

        // Parse bias_comp and imu_status from CORRIMU message
        // bias_comp: true if IMU data is bias compensated (available since Vision-RTK 2 software v2.119.0)
        // imu_status: IMU bias convergence status (see FpaConsts for values)
        bool biasComp = fpa_imu_raw->bias_comp;
        int8_t imuStatus = fpa_imu_raw->imu_status;

        // Log IMU status periodically for debugging (every 500 messages ~ 2.5s at 200Hz)
        ROS_DEBUG("CORRIMU status - bias_comp: %s, imu_status: %d",
                     biasComp ? "true" : "false", imuStatus);

        // Extract sensor_msgs/Imu from FpaImu and apply coordinate transform
        // imuConverter applies extRot to acc and gyro
        sensor_msgs::Imu thisImu = imuConverter(*fpa_imu_raw);

        // Orientation handling:
        // - Before system init: use orientation from /imu/data (fixposition fusion)
        // - After system init: use integrated orientation corrected by LiDAR odometry
        Eigen::Quaterniond orientationToUse;

        if (systemInitialized)
        {
            // After init: integrate from angular velocity, corrected by LiDAR odometry
            if (lastCorrImuTime_ > 0)
            {
                double dt = imuTime - lastCorrImuTime_;
                if (dt > 0 && dt < 1.0)
                {
                    double wx = thisImu.angular_velocity.x;
                    double wy = thisImu.angular_velocity.y;
                    double wz = thisImu.angular_velocity.z;
                    Eigen::Vector3d omega(wx, wy, wz);

                    double angle = omega.norm() * dt;
                    Eigen::Quaterniond q_delta;
                    if (angle > 1e-10)
                    {
                        Eigen::Vector3d axis = omega.normalized();
                        q_delta = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
                    }
                    else
                    {
                        q_delta = Eigen::Quaterniond(1.0, 0.5*omega.x()*dt, 0.5*omega.y()*dt, 0.5*omega.z()*dt);
                        q_delta.normalize();
                    }
                    corrImuOrientation_ = corrImuOrientation_ * q_delta;
                    corrImuOrientation_.normalize();
                }
            }
            orientationToUse = corrImuOrientation_;
        }
        else
        {
            // Before init: use orientation from /imu/data
            if (hasImuOrientation_)
            {
                orientationToUse = latestImuOrientation_;
            }
            else
            {
                orientationToUse = Eigen::Quaterniond::Identity();
            }
        }
        lastCorrImuTime_ = imuTime;

	        // Set orientation in IMU message
	        thisImu.orientation.x = orientationToUse.x();
	        thisImu.orientation.y = orientationToUse.y();
	        thisImu.orientation.z = orientationToUse.z();
	        thisImu.orientation.w = orientationToUse.w();

	        // Debug: print published orientation (first 5 messages only)
	        static int pubDebugCount = 0;
	        if (pubDebugCount < 5 && hasImuOrientation_)
        {
            tf::Quaternion q_pub(orientationToUse.x(), orientationToUse.y(),
                                  orientationToUse.z(), orientationToUse.w());
            double r_pub, p_pub, y_pub;
            tf::Matrix3x3(q_pub).getRPY(r_pub, p_pub, y_pub);
            ROS_WARN("fpaImuHandler PUBLISH #%d: yaw=%.1f deg (systemInit=%s, hasImuOri=%s)",
                     pubDebugCount, y_pub * 180.0 / M_PI,
                     systemInitialized ? "true" : "false",
                     hasImuOrientation_ ? "true" : "false");
            pubDebugCount++;
        }

        // Publish IMU message with integrated orientation for other nodes to use
        // Publish always so imageProjection can get orientation for initialization
        {
            sensor_msgs::Imu imuMsg = thisImu;
            imuMsg.header.frame_id = lidarFrame;
            pubImuWithOrientation.publish(imuMsg);
        }

        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false)
            return;

        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();

        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }

    // Reset CORRIMU orientation when optimization resets or when receiving LiDAR odometry correction
    void resetCorrImuOrientation(const Eigen::Quaterniond& newOrientation)
    {
        corrImuOrientation_ = newOrientation;
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");
    
    IMUPreintegration ImuP;

    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
