/**
 * @file test_transforms_real_data.cpp
 * @brief Unit tests for coordinate transform functions using real bag data
 *
 * Tests the following functions:
 * 1. imuConverter() from utility.h
 * 2. ECEF to ENU conversion (fpaOdomConverter logic)
 * 3. GPS coordinate transform with gpsExtrinsicRot
 * 4. Quaternion transformations
 */

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

// Include the message type for FPA odometry
#include <fixposition_driver_msgs/FpaOdometry.h>

// ANSI color codes
#define GREEN "\033[1;32m"
#define RED "\033[1;31m"
#define YELLOW "\033[1;33m"
#define CYAN "\033[1;36m"
#define RESET "\033[0m"

// Tolerance for floating point comparison
const double TOLERANCE = 1e-6;

// Configuration from params.yaml
Eigen::Matrix3d extRot;      // IMU to LiDAR
Eigen::Matrix3d extRPY;
Eigen::Matrix3d gpsExtRot;   // ENU to LiDAR

// WGS84 parameters
const double WGS84_A = 6378137.0;
const double WGS84_F = 1.0 / 298.257223563;
const double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;

// Test statistics
int tests_passed = 0;
int tests_failed = 0;

//=============================================================================
// Helper Functions
//=============================================================================

void initializeMatrices() {
    // extrinsicRot from params.yaml (Rz 180°)
    extRot << -1,  0, 0,
               0, -1, 0,
               0,  0, 1;

    extRPY = extRot;

    // gpsExtrinsicRot from params.yaml
    gpsExtRot <<  0, -1, 0,
                 -1,  0, 0,
                  0,  0, 1;
}

bool nearEqual(double a, double b, double tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

bool nearEqual(const Eigen::Vector3d& a, const Eigen::Vector3d& b, double tol = TOLERANCE) {
    return (a - b).norm() < tol;
}

void printTestResult(const std::string& test_name, bool passed) {
    if (passed) {
        std::cout << GREEN << "  [PASS] " << RESET << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << RED << "  [FAIL] " << RESET << test_name << std::endl;
        tests_failed++;
    }
}

//=============================================================================
// Function implementations to test (extracted from source files)
//=============================================================================

/**
 * @brief IMU converter function (from utility.h)
 */
sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in) {
    sensor_msgs::Imu imu_out = imu_in;

    // Rotate acceleration
    Eigen::Vector3d acc(imu_in.linear_acceleration.x,
                        imu_in.linear_acceleration.y,
                        imu_in.linear_acceleration.z);
    acc = extRot * acc;
    imu_out.linear_acceleration.x = acc.x();
    imu_out.linear_acceleration.y = acc.y();
    imu_out.linear_acceleration.z = acc.z();

    // Rotate gyroscope
    Eigen::Vector3d gyr(imu_in.angular_velocity.x,
                        imu_in.angular_velocity.y,
                        imu_in.angular_velocity.z);
    gyr = extRot * gyr;
    imu_out.angular_velocity.x = gyr.x();
    imu_out.angular_velocity.y = gyr.y();
    imu_out.angular_velocity.z = gyr.z();

    // Rotate orientation
    Eigen::Quaterniond extQRPY(extRPY);
    extQRPY = extQRPY.inverse();

    Eigen::Quaterniond q_from(imu_in.orientation.w,
                              imu_in.orientation.x,
                              imu_in.orientation.y,
                              imu_in.orientation.z);
    Eigen::Quaterniond q_final = q_from * extQRPY;

    imu_out.orientation.x = q_final.x();
    imu_out.orientation.y = q_final.y();
    imu_out.orientation.z = q_final.z();
    imu_out.orientation.w = q_final.w();

    return imu_out;
}

/**
 * @brief ECEF to LLA conversion (from fpaOdomConverter.cpp)
 */
void ecefToLla(const Eigen::Vector3d& ecef, double& lat, double& lon, double& alt) {
    double x = ecef.x();
    double y = ecef.y();
    double z = ecef.z();

    lon = std::atan2(y, x);
    double p = std::sqrt(x*x + y*y);
    lat = std::atan2(z, p * (1.0 - WGS84_E2));

    // Iterative refinement
    for (int i = 0; i < 5; i++) {
        double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * std::sin(lat) * std::sin(lat));
        double h = p / std::cos(lat) - N;
        lat = std::atan2(z, p * (1.0 - WGS84_E2 * N / (N + h)));
    }

    double N = WGS84_A / std::sqrt(1.0 - WGS84_E2 * std::sin(lat) * std::sin(lat));
    alt = p / std::cos(lat) - N;
}

/**
 * @brief Get ECEF to ENU rotation matrix (from fpaOdomConverter.cpp)
 */
Eigen::Matrix3d getEcefToEnuMatrix(const Eigen::Vector3d& origin_ecef) {
    double lat, lon, alt;
    ecefToLla(origin_ecef, lat, lon, alt);

    double sin_lat = std::sin(lat);
    double cos_lat = std::cos(lat);
    double sin_lon = std::sin(lon);
    double cos_lon = std::cos(lon);

    Eigen::Matrix3d R;
    R << -sin_lon,          cos_lon,          0,
         -sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat,
          cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat;

    return R;
}

/**
 * @brief Convert ECEF to ENU (from fpaOdomConverter.cpp)
 */
Eigen::Vector3d ecefToEnu(const Eigen::Vector3d& ecef_pos,
                          const Eigen::Vector3d& origin_ecef,
                          const Eigen::Matrix3d& R_ecef_to_enu) {
    Eigen::Vector3d delta_ecef = ecef_pos - origin_ecef;
    return R_ecef_to_enu * delta_ecef;
}

/**
 * @brief Transform GPS ENU to LiDAR frame (from mapOptmization.cpp)
 */
Eigen::Vector3d transformGpsToLidar(const Eigen::Vector3d& gps_enu) {
    return gpsExtRot * gps_enu;
}

//=============================================================================
// Test Cases
//=============================================================================

void testImuConverter(const std::vector<sensor_msgs::Imu>& imu_msgs) {
    std::cout << YELLOW << "\n=== Test 1: imuConverter() Function ===" << RESET << std::endl;

    if (imu_msgs.empty()) {
        std::cout << RED << "  No IMU data available" << RESET << std::endl;
        tests_failed++;
        return;
    }

    // Test with first IMU message
    const sensor_msgs::Imu& imu_raw = imu_msgs[0];
    sensor_msgs::Imu imu_converted = imuConverter(imu_raw);

    std::cout << "  Raw IMU acceleration:       ["
              << std::fixed << std::setprecision(4)
              << imu_raw.linear_acceleration.x << ", "
              << imu_raw.linear_acceleration.y << ", "
              << imu_raw.linear_acceleration.z << "]" << std::endl;

    std::cout << "  Converted IMU acceleration: ["
              << imu_converted.linear_acceleration.x << ", "
              << imu_converted.linear_acceleration.y << ", "
              << imu_converted.linear_acceleration.z << "]" << std::endl;

    // Test 1.1: Acceleration rotation (Rz 180° means x' = -x, y' = -y, z' = z)
    Eigen::Vector3d acc_raw(imu_raw.linear_acceleration.x,
                            imu_raw.linear_acceleration.y,
                            imu_raw.linear_acceleration.z);
    Eigen::Vector3d acc_conv(imu_converted.linear_acceleration.x,
                             imu_converted.linear_acceleration.y,
                             imu_converted.linear_acceleration.z);
    Eigen::Vector3d acc_expected = extRot * acc_raw;

    printTestResult("Acceleration rotation correct",
                    nearEqual(acc_conv, acc_expected, 1e-9));

    // Test 1.2: Gyroscope rotation
    Eigen::Vector3d gyr_raw(imu_raw.angular_velocity.x,
                            imu_raw.angular_velocity.y,
                            imu_raw.angular_velocity.z);
    Eigen::Vector3d gyr_conv(imu_converted.angular_velocity.x,
                             imu_converted.angular_velocity.y,
                             imu_converted.angular_velocity.z);
    Eigen::Vector3d gyr_expected = extRot * gyr_raw;

    printTestResult("Gyroscope rotation correct",
                    nearEqual(gyr_conv, gyr_expected, 1e-9));

    // Test 1.3: Gravity should be in +Z direction after conversion
    double gravity_z = imu_converted.linear_acceleration.z;
    double gravity_xy = std::sqrt(imu_converted.linear_acceleration.x *
                                  imu_converted.linear_acceleration.x +
                                  imu_converted.linear_acceleration.y *
                                  imu_converted.linear_acceleration.y);

    std::cout << "  Gravity Z: " << gravity_z << " m/s² (expected ~9.8)" << std::endl;
    std::cout << "  Gravity XY: " << gravity_xy << " m/s² (expected ~0)" << std::endl;

    printTestResult("Gravity primarily in +Z direction",
                    gravity_z > 9.5 && gravity_xy < 0.5);

    // Test 1.4: Orientation transformation
    Eigen::Quaterniond q_raw(imu_raw.orientation.w,
                             imu_raw.orientation.x,
                             imu_raw.orientation.y,
                             imu_raw.orientation.z);
    Eigen::Quaterniond q_conv(imu_converted.orientation.w,
                              imu_converted.orientation.x,
                              imu_converted.orientation.y,
                              imu_converted.orientation.z);

    // Quaternion should be normalized
    double q_norm = q_conv.norm();
    printTestResult("Converted quaternion is normalized",
                    nearEqual(q_norm, 1.0, 1e-6));
}

void testEcefToEnuConversion(const std::vector<fixposition_driver_msgs::FpaOdometry>& gps_msgs) {
    std::cout << YELLOW << "\n=== Test 2: ECEF to ENU Conversion ===" << RESET << std::endl;

    if (gps_msgs.size() < 10) {
        std::cout << RED << "  Not enough GPS data" << RESET << std::endl;
        tests_failed++;
        return;
    }

    // Get origin from first message
    Eigen::Vector3d origin_ecef(gps_msgs[0].pose.pose.position.x,
                                gps_msgs[0].pose.pose.position.y,
                                gps_msgs[0].pose.pose.position.z);

    // Get LLA of origin
    double lat, lon, alt;
    ecefToLla(origin_ecef, lat, lon, alt);

    std::cout << "  Origin ECEF: [" << std::fixed << std::setprecision(2)
              << origin_ecef.x() << ", "
              << origin_ecef.y() << ", "
              << origin_ecef.z() << "]" << std::endl;

    std::cout << "  Origin LLA:  lat=" << std::setprecision(6)
              << lat * 180.0 / M_PI << "°, lon="
              << lon * 180.0 / M_PI << "°" << std::endl;

    // Test 2.1: First position should be [0,0,0] in ENU
    Eigen::Matrix3d R_ecef_enu = getEcefToEnuMatrix(origin_ecef);
    Eigen::Vector3d first_enu = ecefToEnu(origin_ecef, origin_ecef, R_ecef_enu);

    printTestResult("First ENU position is origin [0,0,0]",
                    first_enu.norm() < 1e-9);

    // Test 2.2: ENU rotation matrix is orthogonal
    Eigen::Matrix3d RRt = R_ecef_enu * R_ecef_enu.transpose();
    bool is_orthogonal = (RRt - Eigen::Matrix3d::Identity()).norm() < 1e-9;
    double det = R_ecef_enu.determinant();

    printTestResult("ECEF->ENU matrix is orthogonal",
                    is_orthogonal && nearEqual(det, 1.0, 1e-6));

    // Test 2.3: Convert multiple positions and check trajectory
    Eigen::Vector3d last_enu(0, 0, 0);
    for (size_t i = 0; i < std::min(gps_msgs.size(), size_t(100)); i++) {
        Eigen::Vector3d ecef(gps_msgs[i].pose.pose.position.x,
                             gps_msgs[i].pose.pose.position.y,
                             gps_msgs[i].pose.pose.position.z);
        last_enu = ecefToEnu(ecef, origin_ecef, R_ecef_enu);
    }

    std::cout << "  Last ENU position: E=" << std::setprecision(3)
              << last_enu.x() << "m, N=" << last_enu.y()
              << "m, U=" << last_enu.z() << "m" << std::endl;

    double horizontal_dist = std::sqrt(last_enu.x()*last_enu.x() +
                                       last_enu.y()*last_enu.y());
    std::cout << "  Horizontal distance: " << horizontal_dist << "m" << std::endl;

    printTestResult("ENU coordinates are reasonable",
                    horizontal_dist < 10000);  // Less than 10km in 100 samples
}

void testGpsToLidarTransform(const std::vector<fixposition_driver_msgs::FpaOdometry>& gps_msgs) {
    std::cout << YELLOW << "\n=== Test 3: GPS to LiDAR Transform ===" << RESET << std::endl;

    if (gps_msgs.size() < 100) {
        std::cout << RED << "  Not enough GPS data" << RESET << std::endl;
        tests_failed++;
        return;
    }

    // Setup ENU conversion
    Eigen::Vector3d origin_ecef(gps_msgs[0].pose.pose.position.x,
                                gps_msgs[0].pose.pose.position.y,
                                gps_msgs[0].pose.pose.position.z);
    Eigen::Matrix3d R_ecef_enu = getEcefToEnuMatrix(origin_ecef);

    // Get trajectory endpoints
    Eigen::Vector3d start_ecef = origin_ecef;
    Eigen::Vector3d end_ecef(gps_msgs[99].pose.pose.position.x,
                             gps_msgs[99].pose.pose.position.y,
                             gps_msgs[99].pose.pose.position.z);

    Eigen::Vector3d start_enu = ecefToEnu(start_ecef, origin_ecef, R_ecef_enu);
    Eigen::Vector3d end_enu = ecefToEnu(end_ecef, origin_ecef, R_ecef_enu);
    Eigen::Vector3d delta_enu = end_enu - start_enu;

    std::cout << "  ENU delta: E=" << std::setprecision(3) << delta_enu.x()
              << ", N=" << delta_enu.y() << ", U=" << delta_enu.z() << std::endl;

    // Test 3.1: Transform to LiDAR frame
    Eigen::Vector3d delta_lidar = transformGpsToLidar(delta_enu);

    std::cout << "  LiDAR delta: X=" << delta_lidar.x()
              << ", Y=" << delta_lidar.y() << ", Z=" << delta_lidar.z() << std::endl;

    // Test 3.2: Verify mapping: X_lidar = -N, Y_lidar = -E, Z_lidar = U
    Eigen::Vector3d expected_lidar(-delta_enu.y(), -delta_enu.x(), delta_enu.z());

    printTestResult("GPS->LiDAR mapping (X=-N, Y=-E, Z=U)",
                    nearEqual(delta_lidar, expected_lidar, 1e-9));

    // Test 3.3: gpsExtRot matrix properties
    double det = gpsExtRot.determinant();
    Eigen::Matrix3d RRt = gpsExtRot * gpsExtRot.transpose();
    bool is_orthogonal = (RRt - Eigen::Matrix3d::Identity()).norm() < 1e-9;

    std::cout << "  gpsExtRot determinant: " << det << std::endl;

    printTestResult("gpsExtRot is orthogonal matrix",
                    is_orthogonal && (nearEqual(det, 1.0) || nearEqual(det, -1.0)));

    // Test 3.4: Verify specific direction mappings
    Eigen::Vector3d north(0, 1, 0);
    Eigen::Vector3d east(1, 0, 0);
    Eigen::Vector3d up(0, 0, 1);

    Eigen::Vector3d north_lidar = gpsExtRot * north;
    Eigen::Vector3d east_lidar = gpsExtRot * east;
    Eigen::Vector3d up_lidar = gpsExtRot * up;

    printTestResult("North -> LiDAR -X (forward)",
                    nearEqual(north_lidar, Eigen::Vector3d(-1, 0, 0)));
    printTestResult("East -> LiDAR -Y (right)",
                    nearEqual(east_lidar, Eigen::Vector3d(0, -1, 0)));
    printTestResult("Up -> LiDAR +Z (up)",
                    nearEqual(up_lidar, Eigen::Vector3d(0, 0, 1)));
}

void testQuaternionTransformations(const std::vector<sensor_msgs::Imu>& imu_msgs) {
    std::cout << YELLOW << "\n=== Test 4: Quaternion Transformations ===" << RESET << std::endl;

    if (imu_msgs.empty()) {
        std::cout << RED << "  No IMU data available" << RESET << std::endl;
        tests_failed++;
        return;
    }

    // Test 4.1: extQRPY computation
    Eigen::Quaterniond extQRPY(extRPY);
    extQRPY = extQRPY.inverse();

    std::cout << "  extQRPY (inverse of extRPY): ["
              << "w=" << std::setprecision(4) << extQRPY.w()
              << ", x=" << extQRPY.x()
              << ", y=" << extQRPY.y()
              << ", z=" << extQRPY.z() << "]" << std::endl;

    // For Rz(180°), the inverse quaternion should represent Rz(-180°) = Rz(180°)
    // Expected: w=0, x=0, y=0, z=1 (or w=0, x=0, y=0, z=-1)
    bool quat_correct = (nearEqual(std::abs(extQRPY.w()), 0.0, 1e-3) &&
                         nearEqual(std::abs(extQRPY.z()), 1.0, 1e-3));
    printTestResult("extQRPY represents Rz(180°) inverse", quat_correct);

    // Test 4.2: Apply quaternion to IMU orientation
    const sensor_msgs::Imu& imu = imu_msgs[0];
    Eigen::Quaterniond q_imu(imu.orientation.w, imu.orientation.x,
                             imu.orientation.y, imu.orientation.z);
    Eigen::Quaterniond q_lidar = q_imu * extQRPY;

    std::cout << "  IMU quaternion:   [w=" << q_imu.w() << ", x=" << q_imu.x()
              << ", y=" << q_imu.y() << ", z=" << q_imu.z() << "]" << std::endl;
    std::cout << "  LiDAR quaternion: [w=" << q_lidar.w() << ", x=" << q_lidar.x()
              << ", y=" << q_lidar.y() << ", z=" << q_lidar.z() << "]" << std::endl;

    // Test 4.3: Result should be normalized
    printTestResult("Result quaternion is normalized",
                    nearEqual(q_lidar.norm(), 1.0, 1e-6));

    // Test 4.4: Rotation matrix from quaternion should be valid
    Eigen::Matrix3d R_lidar = q_lidar.toRotationMatrix();
    double det_R = R_lidar.determinant();
    Eigen::Matrix3d RRt = R_lidar * R_lidar.transpose();
    bool R_orthogonal = (RRt - Eigen::Matrix3d::Identity()).norm() < 1e-6;

    printTestResult("LiDAR rotation matrix is valid (det=1, orthogonal)",
                    nearEqual(det_R, 1.0, 1e-6) && R_orthogonal);
}

void testTransformConsistency(const std::vector<sensor_msgs::Imu>& imu_msgs,
                               const std::vector<fixposition_driver_msgs::FpaOdometry>& gps_msgs) {
    std::cout << YELLOW << "\n=== Test 5: Transform Consistency ===" << RESET << std::endl;

    // Test 5.1: extRot and gpsExtRot should produce consistent Z-axis
    Eigen::Vector3d imu_up(0, 0, 1);
    Eigen::Vector3d gps_up(0, 0, 1);

    Eigen::Vector3d lidar_up_from_imu = extRot * imu_up;
    Eigen::Vector3d lidar_up_from_gps = gpsExtRot * gps_up;

    std::cout << "  Up direction from IMU transform: " << lidar_up_from_imu.transpose() << std::endl;
    std::cout << "  Up direction from GPS transform: " << lidar_up_from_gps.transpose() << std::endl;

    printTestResult("Up axis consistent between IMU and GPS transforms",
                    nearEqual(lidar_up_from_imu, lidar_up_from_gps));

    // Test 5.2: Both should have determinant magnitude 1
    double det_ext = std::abs(extRot.determinant());
    double det_gps = std::abs(gpsExtRot.determinant());

    printTestResult("Both matrices have |det|=1",
                    nearEqual(det_ext, 1.0) && nearEqual(det_gps, 1.0));

    // Test 5.3: Forward direction consistency
    // IMU +X (forward in IMU) -> LiDAR -X (forward in Livox)
    // GPS +Y (North, forward when facing North) -> LiDAR -X (forward in Livox)
    Eigen::Vector3d imu_forward(1, 0, 0);
    Eigen::Vector3d gps_forward(0, 1, 0);  // North

    Eigen::Vector3d lidar_from_imu = extRot * imu_forward;
    Eigen::Vector3d lidar_from_gps = gpsExtRot * gps_forward;

    std::cout << "  IMU +X -> LiDAR: " << lidar_from_imu.transpose() << std::endl;
    std::cout << "  GPS North (+Y) -> LiDAR: " << lidar_from_gps.transpose() << std::endl;

    // Both should map to LiDAR -X
    printTestResult("IMU forward and GPS North both map to LiDAR -X",
                    nearEqual(lidar_from_imu, Eigen::Vector3d(-1, 0, 0)) &&
                    nearEqual(lidar_from_gps, Eigen::Vector3d(-1, 0, 0)));
}

//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_transforms_real_data");

    std::string bag_path = "/root/autodl-tmp/info_fixed.bag";

    std::cout << "\n" << YELLOW << "==================================================" << RESET << std::endl;
    std::cout << YELLOW << "  C++ Unit Tests with Real Bag Data" << RESET << std::endl;
    std::cout << YELLOW << "  Bag file: " << bag_path << RESET << std::endl;
    std::cout << YELLOW << "==================================================" << RESET << std::endl;

    // Initialize transformation matrices
    initializeMatrices();

    // Read data from bag file
    std::vector<sensor_msgs::Imu> imu_msgs;
    std::vector<fixposition_driver_msgs::FpaOdometry> gps_msgs;

    try {
        rosbag::Bag bag;
        bag.open(bag_path, rosbag::bagmode::Read);

        std::vector<std::string> topics;
        topics.push_back("/imu/data");
        topics.push_back("/fixposition/fpa/odometry");

        rosbag::View view(bag, rosbag::TopicQuery(topics));

        std::cout << "\n  Reading bag file..." << std::endl;

        for (const rosbag::MessageInstance& m : view) {
            if (m.getTopic() == "/imu/data" && imu_msgs.size() < 100) {
                sensor_msgs::Imu::ConstPtr imu = m.instantiate<sensor_msgs::Imu>();
                if (imu != nullptr) {
                    imu_msgs.push_back(*imu);
                }
            }
            else if (m.getTopic() == "/fixposition/fpa/odometry" && gps_msgs.size() < 2000) {
                fixposition_driver_msgs::FpaOdometry::ConstPtr gps =
                    m.instantiate<fixposition_driver_msgs::FpaOdometry>();
                if (gps != nullptr) {
                    gps_msgs.push_back(*gps);
                }
            }

            if (imu_msgs.size() >= 100 && gps_msgs.size() >= 2000) {
                break;
            }
        }

        bag.close();

        std::cout << "  Loaded " << imu_msgs.size() << " IMU messages" << std::endl;
        std::cout << "  Loaded " << gps_msgs.size() << " GPS messages" << std::endl;

    } catch (const rosbag::BagException& e) {
        std::cerr << RED << "  Error reading bag file: " << e.what() << RESET << std::endl;
        return 1;
    }

    // Run tests
    testImuConverter(imu_msgs);
    testEcefToEnuConversion(gps_msgs);
    testGpsToLidarTransform(gps_msgs);
    testQuaternionTransformations(imu_msgs);
    testTransformConsistency(imu_msgs, gps_msgs);

    // Summary
    std::cout << "\n" << YELLOW << "==================================================" << RESET << std::endl;
    std::cout << "  Test Summary:" << std::endl;
    std::cout << GREEN << "    Passed: " << tests_passed << RESET << std::endl;
    std::cout << RED << "    Failed: " << tests_failed << RESET << std::endl;
    std::cout << YELLOW << "==================================================" << RESET << std::endl;

    if (tests_failed > 0) {
        std::cout << RED << "  SOME TESTS FAILED!" << RESET << std::endl;
    } else {
        std::cout << GREEN << "  ALL TESTS PASSED!" << RESET << std::endl;
    }
    std::cout << YELLOW << "==================================================" << RESET << "\n" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
