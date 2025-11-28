/**
 * Debug test for coordinate transformations
 * This test loads parameters and verifies the transformation matrices
 */

#include <ros/ros.h>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_coordinate_debug");
    ros::NodeHandle nh;

    // Load parameters
    std::vector<double> extRotV, extRPYV, extTransV;
    nh.param<std::vector<double>>("lio_sam/extrinsicRot", extRotV, std::vector<double>());
    nh.param<std::vector<double>>("lio_sam/extrinsicRPY", extRPYV, std::vector<double>());
    nh.param<std::vector<double>>("lio_sam/extrinsicTrans", extTransV, std::vector<double>());

    float imuGravity;
    nh.param<float>("lio_sam/imuGravity", imuGravity, 9.80511);

    std::cout << "=== Parameter Debug ===" << std::endl;
    std::cout << "imuGravity: " << imuGravity << std::endl;

    std::cout << "\nextRotV size: " << extRotV.size() << std::endl;
    std::cout << "extRPYV size: " << extRPYV.size() << std::endl;
    std::cout << "extTransV size: " << extTransV.size() << std::endl;

    if (extRotV.size() != 9 || extRPYV.size() != 9 || extTransV.size() != 3) {
        std::cerr << "ERROR: Invalid parameter sizes!" << std::endl;
        return 1;
    }

    Eigen::Matrix3d extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
    Eigen::Matrix3d extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
    Eigen::Vector3d extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);

    std::cout << "\nextRot (extrinsicRot):\n" << extRot << std::endl;
    std::cout << "\nextRPY (extrinsicRPY):\n" << extRPY << std::endl;
    std::cout << "\nextTrans (extrinsicTrans): " << extTrans.transpose() << std::endl;

    // Check if it's Rz(180)
    Eigen::Matrix3d Rz180;
    Rz180 << -1, 0, 0,
              0, -1, 0,
              0, 0, 1;

    std::cout << "\n=== Verification ===" << std::endl;
    std::cout << "Expected Rz(180):\n" << Rz180 << std::endl;
    std::cout << "extRot == Rz(180): " << ((extRot - Rz180).norm() < 1e-6 ? "YES" : "NO") << std::endl;
    std::cout << "extRPY == Rz(180): " << ((extRPY - Rz180).norm() < 1e-6 ? "YES" : "NO") << std::endl;

    // Compute extQRPY
    Eigen::Quaterniond extQRPY = Eigen::Quaterniond(extRPY).inverse();
    std::cout << "\nextQRPY (quaternion): ["
              << extQRPY.w() << ", "
              << extQRPY.x() << ", "
              << extQRPY.y() << ", "
              << extQRPY.z() << "]" << std::endl;

    // Test imuConverter transformation
    std::cout << "\n=== IMU Converter Test ===" << std::endl;

    // Test case: stationary, level, facing north (in ENU)
    // IMU frame: +X forward (north), +Y left (west), +Z up
    // When stationary and level, accelerometer reads [0, 0, +g]
    Eigen::Vector3d acc_imu(0, 0, imuGravity);
    Eigen::Vector3d acc_lidar = extRot * acc_imu;
    std::cout << "Acc in IMU frame: " << acc_imu.transpose() << std::endl;
    std::cout << "Acc in LiDAR frame: " << acc_lidar.transpose() << std::endl;
    std::cout << "Expected (stationary, level): [0, 0, " << imuGravity << "]" << std::endl;

    // Test orientation transformation
    // If IMU is level (identity rotation), what is LiDAR orientation?
    Eigen::Quaterniond q_imu_identity(1, 0, 0, 0);  // Identity
    Eigen::Quaterniond q_lidar = q_imu_identity * extQRPY;
    std::cout << "\nIMU orientation (identity): ["
              << q_imu_identity.w() << ", "
              << q_imu_identity.x() << ", "
              << q_imu_identity.y() << ", "
              << q_imu_identity.z() << "]" << std::endl;
    std::cout << "LiDAR orientation after conversion: ["
              << q_lidar.w() << ", "
              << q_lidar.x() << ", "
              << q_lidar.y() << ", "
              << q_lidar.z() << "]" << std::endl;

    // Convert quaternion to Euler angles (roll, pitch, yaw)
    Eigen::Vector3d euler_imu = q_imu_identity.toRotationMatrix().eulerAngles(0, 1, 2);
    Eigen::Vector3d euler_lidar = q_lidar.toRotationMatrix().eulerAngles(0, 1, 2);
    std::cout << "\nIMU Euler (r,p,y) degrees: " << euler_imu.transpose() * 180.0 / M_PI << std::endl;
    std::cout << "LiDAR Euler (r,p,y) degrees: " << euler_lidar.transpose() * 180.0 / M_PI << std::endl;

    // Test gravity compensation in preintegration
    std::cout << "\n=== Gravity Compensation Test ===" << std::endl;
    Eigen::Vector3d g_nav(0, 0, -imuGravity);  // Gravity in nav frame (Z-up convention)

    // When stationary with level orientation:
    // a_true_nav = R_nav_body * (a_body - bias) + g_nav
    // For level orientation, R_nav_body = identity (ignoring yaw)
    Eigen::Matrix3d R_nav_body = Eigen::Matrix3d::Identity();
    Eigen::Vector3d bias(0, 0, 0);
    Eigen::Vector3d a_true_nav = R_nav_body * (acc_lidar - bias) + g_nav;

    std::cout << "Gravity in nav frame: " << g_nav.transpose() << std::endl;
    std::cout << "Accelerometer reading (LiDAR frame): " << acc_lidar.transpose() << std::endl;
    std::cout << "a_true_nav = R * (a - bias) + g = " << a_true_nav.transpose() << std::endl;
    std::cout << "Expected (stationary): [0, 0, 0]" << std::endl;

    if (a_true_nav.norm() < 0.01) {
        std::cout << "\nGravity compensation: CORRECT" << std::endl;
    } else {
        std::cout << "\nGravity compensation: ERROR - " << a_true_nav.norm() << " m/s^2 residual" << std::endl;
    }

    // Test with tilted orientation
    std::cout << "\n=== Tilted Orientation Test ===" << std::endl;
    double pitch_deg = 10.0;
    double pitch_rad = pitch_deg * M_PI / 180.0;

    // IMU tilted by pitch angle
    Eigen::AngleAxisd pitch_rot(pitch_rad, Eigen::Vector3d::UnitY());
    Eigen::Quaterniond q_imu_tilted(pitch_rot);

    // When tilted, accelerometer reads gravity projected onto body axes
    Eigen::Vector3d g_world(0, 0, -imuGravity);
    Eigen::Matrix3d R_body_world = q_imu_tilted.inverse().toRotationMatrix();
    Eigen::Vector3d acc_imu_tilted = -R_body_world * g_world;
    Eigen::Vector3d acc_lidar_tilted = extRot * acc_imu_tilted;

    // Convert orientation
    Eigen::Quaterniond q_lidar_tilted = q_imu_tilted * extQRPY;

    std::cout << "IMU pitch: " << pitch_deg << " degrees" << std::endl;
    std::cout << "Acc in tilted IMU frame: " << acc_imu_tilted.transpose() << std::endl;
    std::cout << "Acc in tilted LiDAR frame: " << acc_lidar_tilted.transpose() << std::endl;

    // Gravity compensation
    Eigen::Matrix3d R_nav_body_tilted = q_lidar_tilted.toRotationMatrix();
    Eigen::Vector3d a_true_tilted = R_nav_body_tilted * (acc_lidar_tilted - bias) + g_nav;

    std::cout << "a_true_nav (tilted) = " << a_true_tilted.transpose() << std::endl;
    std::cout << "Expected (stationary, tilted): [0, 0, 0]" << std::endl;

    if (a_true_tilted.norm() < 0.01) {
        std::cout << "\nTilted gravity compensation: CORRECT" << std::endl;
    } else {
        std::cout << "\nTilted gravity compensation: ERROR - " << a_true_tilted.norm() << " m/s^2 residual" << std::endl;
    }

    return 0;
}
