#include <gtest/gtest.h>

#include <limits>

#include "covariance_utils.h"

// These tests focus on the small utility helpers used by mapOptmization::addGPSFactor():
// - rotating full 6x6 covariances between frames
// - reordering ROS pose covariance layout to GTSAM Pose3 tangent layout
// - enforcing SPD to prevent ISAM2 failures on malformed covariances

namespace
{

Eigen::Matrix3d rotZ90()
{
    Eigen::Matrix3d R;
    R << 0.0, -1.0, 0.0,
         1.0, 0.0, 0.0,
         0.0, 0.0, 1.0;
    return R;
}

Eigen::Matrix3d rotX90()
{
    Eigen::Matrix3d R;
    // +90deg about X: y->-z, z->y
    R << 1.0, 0.0, 0.0,
         0.0, 0.0, -1.0,
         0.0, 1.0, 0.0;
    return R;
}

} // namespace

TEST(CovarianceUtils, RotateCov6SwapsXY)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov(0, 0) = 1.0;
    cov(1, 1) = 4.0;
    cov(2, 2) = 9.0;
    cov(3, 3) = 0.01;
    cov(4, 4) = 0.04;
    cov(5, 5) = 0.09;

    const Eigen::Matrix<double, 6, 6> cov_rot = lio_sam::cov::rotateCov6(cov, rotZ90());

    EXPECT_NEAR(cov_rot(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(cov_rot(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(cov_rot(2, 2), 9.0, 1e-12);

    EXPECT_NEAR(cov_rot(3, 3), 0.04, 1e-12);
    EXPECT_NEAR(cov_rot(4, 4), 0.01, 1e-12);
    EXPECT_NEAR(cov_rot(5, 5), 0.09, 1e-12);
}

TEST(CovarianceUtils, RotateCov6RotatesXYOffDiagonal)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
    cov(0, 0) = 1.0;
    cov(1, 1) = 4.0;
    cov(0, 1) = 0.5;
    cov(1, 0) = 0.5;
    cov(2, 2) = 9.0;

    const Eigen::Matrix<double, 6, 6> cov_rot = lio_sam::cov::rotateCov6(cov, rotZ90());

    // 90deg rotation swaps x/y and flips the sign of xy correlation.
    EXPECT_NEAR(cov_rot(0, 0), 4.0, 1e-12);
    EXPECT_NEAR(cov_rot(1, 1), 1.0, 1e-12);
    EXPECT_NEAR(cov_rot(0, 1), -0.5, 1e-12);
    EXPECT_NEAR(cov_rot(1, 0), -0.5, 1e-12);
    EXPECT_NEAR(cov_rot(2, 2), 9.0, 1e-12);
    EXPECT_TRUE((cov_rot.isApprox(cov_rot.transpose(), 1e-12)));
}

TEST(CovarianceUtils, RotateCov6RotatesCrossBlock)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();

    Eigen::Matrix3d pos;
    pos << 1.0, 0.2, -0.1,
           0.2, 2.0, 0.3,
          -0.1, 0.3, 3.0;

    Eigen::Matrix3d ori;
    ori << 0.01, 0.001, 0.0,
           0.001, 0.02, -0.002,
           0.0, -0.002, 0.03;

    Eigen::Matrix3d cross;
    cross << 0.1, 0.2, 0.3,
             -0.4, 0.5, 0.6,
             0.7, -0.8, 0.9;

    cov.block<3, 3>(0, 0) = pos;
    cov.block<3, 3>(3, 3) = ori;
    cov.block<3, 3>(0, 3) = cross;
    cov.block<3, 3>(3, 0) = cross.transpose();

    const Eigen::Matrix3d R = rotZ90();
    const Eigen::Matrix<double, 6, 6> cov_rot = lio_sam::cov::rotateCov6(cov, R);

    const Eigen::Matrix3d pos_expected = R * pos * R.transpose();
    const Eigen::Matrix3d ori_expected = R * ori * R.transpose();
    const Eigen::Matrix3d cross_expected = R * cross * R.transpose();

    EXPECT_TRUE((cov_rot.isApprox(cov_rot.transpose(), 1e-12)));
    EXPECT_TRUE((cov_rot.block<3, 3>(0, 0).isApprox(pos_expected, 1e-12)));
    EXPECT_TRUE((cov_rot.block<3, 3>(3, 3).isApprox(ori_expected, 1e-12)));
    EXPECT_TRUE((cov_rot.block<3, 3>(0, 3).isApprox(cross_expected, 1e-12)));
    EXPECT_TRUE((cov_rot.block<3, 3>(3, 0).isApprox(cross_expected.transpose(), 1e-12)));
}

TEST(CovarianceUtils, ReorderRosPoseCovToGtsam)
{
    Eigen::Matrix<double, 6, 6> cov_ros = Eigen::Matrix<double, 6, 6>::Zero();

    Eigen::Matrix3d pos;
    pos << 1.0, 0.1, 0.2,
           0.1, 2.0, 0.3,
           0.2, 0.3, 3.0;

    Eigen::Matrix3d ori;
    ori << 4.0, 0.4, 0.5,
           0.4, 5.0, 0.6,
           0.5, 0.6, 6.0;

    Eigen::Matrix3d cross_pos_ori;
    cross_pos_ori << 0.01, 0.02, 0.03,
                     0.04, 0.05, 0.06,
                     0.07, 0.08, 0.09;

    cov_ros.block<3, 3>(0, 0) = pos;
    cov_ros.block<3, 3>(3, 3) = ori;
    cov_ros.block<3, 3>(0, 3) = cross_pos_ori;
    cov_ros.block<3, 3>(3, 0) = cross_pos_ori.transpose();

    const Eigen::Matrix<double, 6, 6> cov_gtsam = lio_sam::cov::reorderRosPoseCovToGtsam(cov_ros);

    EXPECT_TRUE((cov_gtsam.isApprox(cov_gtsam.transpose(), 1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(0, 0).isApprox(ori, 1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(3, 3).isApprox(pos, 1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(0, 3).isApprox(cross_pos_ori.transpose(), 1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(3, 0).isApprox(cross_pos_ori, 1e-12)));
}

TEST(CovarianceUtils, MakeOrientationOnlyPoseCovGtsamYawOnly)
{
    Eigen::Matrix<double, 6, 6> cov_ros = Eigen::Matrix<double, 6, 6>::Zero();
    cov_ros(0, 0) = 1.0;
    cov_ros(1, 1) = 2.0;
    cov_ros(2, 2) = 3.0;
    cov_ros(3, 3) = 0.01;
    cov_ros(4, 4) = 0.02;
    cov_ros(5, 5) = 0.03; // yaw variance in ROS layout

    // Add some correlations that should be removed.
    cov_ros(0, 3) = 0.5;
    cov_ros(3, 0) = 0.4;
    cov_ros(1, 2) = 0.7;
    cov_ros(2, 1) = 0.6;

    const double weakVar = 123.0;
    const double hugePosVar = 456.0;
    const Eigen::Matrix<double, 6, 6> cov_gtsam =
        lio_sam::cov::makeOrientationOnlyPoseCovGtsam(cov_ros, /*yaw_only=*/true, weakVar, hugePosVar);

    EXPECT_TRUE((cov_gtsam.isApprox(cov_gtsam.transpose(), 1e-12)));

    // Rotation block: roll/pitch weak, yaw kept.
    EXPECT_NEAR(cov_gtsam(0, 0), weakVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(1, 1), weakVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(2, 2), cov_ros(5, 5), 1e-12);

    // Translation block: huge variances, no correlations.
    EXPECT_NEAR(cov_gtsam(3, 3), hugePosVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(4, 4), hugePosVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(5, 5), hugePosVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(3, 4), 0.0, 1e-12);
    EXPECT_NEAR(cov_gtsam(4, 5), 0.0, 1e-12);
    EXPECT_NEAR(cov_gtsam(3, 5), 0.0, 1e-12);

    // Rot-trans blocks must be zero.
    EXPECT_TRUE((cov_gtsam.block<3, 3>(0, 3).isZero(1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(3, 0).isZero(1e-12)));

    // Yaw should be independent from roll/pitch.
    EXPECT_NEAR(cov_gtsam(0, 2), 0.0, 1e-12);
    EXPECT_NEAR(cov_gtsam(1, 2), 0.0, 1e-12);
    EXPECT_NEAR(cov_gtsam(2, 0), 0.0, 1e-12);
    EXPECT_NEAR(cov_gtsam(2, 1), 0.0, 1e-12);
}

TEST(CovarianceUtils, MakeOrientationOnlyPoseCovGtsamKeepsRotBlockWhenNotYawOnly)
{
    Eigen::Matrix<double, 6, 6> cov_ros = Eigen::Matrix<double, 6, 6>::Zero();

    Eigen::Matrix3d ori;
    ori << 0.01, 0.002, 0.003,
           0.001, 0.02, 0.004,
           0.005, 0.006, 0.03;
    cov_ros.block<3, 3>(3, 3) = ori;

    const double weakVar = 1e3;
    const double hugePosVar = 1e6;
    const Eigen::Matrix<double, 6, 6> cov_gtsam =
        lio_sam::cov::makeOrientationOnlyPoseCovGtsam(cov_ros, /*yaw_only=*/false, weakVar, hugePosVar);

    const Eigen::Matrix3d ori_expected = (0.5 * (ori + ori.transpose())).eval();
    EXPECT_TRUE((cov_gtsam.block<3, 3>(0, 0).isApprox(ori_expected, 1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(0, 3).isZero(1e-12)));
    EXPECT_TRUE((cov_gtsam.block<3, 3>(3, 0).isZero(1e-12)));
    EXPECT_NEAR(cov_gtsam(3, 3), hugePosVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(4, 4), hugePosVar, 1e-12);
    EXPECT_NEAR(cov_gtsam(5, 5), hugePosVar, 1e-12);
}

TEST(CovarianceUtils, SanitizeCovariance6ReplacesNonFiniteAndSymmetrizes)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov(0, 1) = 0.2;
    cov(1, 0) = 0.1; // asymmetric on purpose
    cov(2, 2) = std::numeric_limits<double>::infinity();

    ASSERT_TRUE(lio_sam::cov::sanitizeCovariance(&cov));
    EXPECT_TRUE((cov.isApprox(cov.transpose(), 1e-12)));
    EXPECT_NEAR(cov(0, 1), 0.15, 1e-12);
    EXPECT_NEAR(cov(1, 0), 0.15, 1e-12);
    EXPECT_DOUBLE_EQ(cov(2, 2), 0.0);
}

TEST(CovarianceUtils, SanitizeCovariance6NoChangeWhenFinite)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov(0, 1) = 0.05;
    cov(1, 0) = 0.05;

    Eigen::Matrix<double, 6, 6> cov_in = cov;
    ASSERT_FALSE(lio_sam::cov::sanitizeCovariance(&cov));
    EXPECT_TRUE((cov.isApprox(cov_in, 1e-12)));
}

TEST(CovarianceUtils, SanitizeCovariance3ReplacesNonFiniteAndSymmetrizes)
{
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    cov(0, 1) = 2.0;
    cov(1, 0) = 4.0;
    cov(2, 2) = std::numeric_limits<double>::quiet_NaN();

    ASSERT_TRUE(lio_sam::cov::sanitizeCovariance(&cov));
    EXPECT_TRUE((cov.isApprox(cov.transpose(), 1e-12)));
    EXPECT_NEAR(cov(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(cov(1, 0), 3.0, 1e-12);
    EXPECT_DOUBLE_EQ(cov(2, 2), 0.0);
}

TEST(CovarianceUtils, DisableElevationAfterRotationAvoidsLeakage)
{
    // Regression test for a subtle frame-mixing bug:
    // If you "disable elevation" by inflating ENU-z covariance BEFORE rotating ENU->LiDAR,
    // and the rotation mixes z with x/y (i.e. has roll/pitch), the huge variance leaks into x/y.
    // Correct behavior: rotate first, then inflate LiDAR-z covariance.

    Eigen::Matrix<double, 6, 6> cov_ros = Eigen::Matrix<double, 6, 6>::Zero();
    cov_ros(0, 0) = 1.0;  // x
    cov_ros(1, 1) = 4.0;  // y
    cov_ros(2, 2) = 9.0;  // z
    cov_ros(3, 3) = 0.01; // roll
    cov_ros(4, 4) = 0.01; // pitch
    cov_ros(5, 5) = 0.01; // yaw

    const double scale = 0.05;
    const double zVarHuge = 1e6 * scale;

    // A rotation that mixes y/z (simulating a gpsExtRot with roll/pitch).
    const Eigen::Matrix3d R = rotX90();

    // Wrong/old way: inflate ENU-z then rotate -> leaks into LiDAR-y due to y/z mixing.
    Eigen::Matrix<double, 6, 6> cov_old = cov_ros;
    cov_old.row(2).setZero();
    cov_old.col(2).setZero();
    cov_old(2, 2) = 1e6;
    cov_old *= scale;
    const Eigen::Matrix<double, 6, 6> cov_old_rot = lio_sam::cov::rotateCov6(cov_old, R);

    // Correct/new way: rotate first, then inflate LiDAR-z.
    const Eigen::Matrix<double, 6, 6> cov_rot = lio_sam::cov::rotateCov6(cov_ros * scale, R);
    Eigen::Matrix<double, 6, 6> cov_new = cov_rot;
    cov_new.row(2).setZero();
    cov_new.col(2).setZero();
    cov_new(2, 2) = zVarHuge;

    // R mixes y/z so LiDAR-y variance should come from ENU-z (9*scale).
    EXPECT_NEAR(cov_new(1, 1), 9.0 * scale, 1e-12);
    EXPECT_NEAR(cov_new(2, 2), zVarHuge, 1e-6);

    // Old behavior: LiDAR-y variance incorrectly becomes huge.
    EXPECT_GT(cov_old_rot(1, 1), 1e4);
}

TEST(CovarianceUtils, ProjectCovariance6ToSPD)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov(5, 5) = -0.1;

    const double eps = 1e-6;
    double minEig = 0.0;
    ASSERT_TRUE(lio_sam::cov::projectCovarianceToSPD(&cov, eps, &minEig));
    EXPECT_LT(minEig, eps);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(cov);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), eps * 0.999);
}

TEST(CovarianceUtils, ProjectCovariance6SymmetrizesInput)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov(0, 1) = 0.2;
    cov(1, 0) = 0.1; // asymmetric on purpose

    const double eps = 1e-6;
    ASSERT_TRUE(lio_sam::cov::projectCovarianceToSPD(&cov, eps));
    EXPECT_TRUE((cov.isApprox(cov.transpose(), 1e-12)));
}

TEST(CovarianceUtils, ProjectCovariance6LeavesSPDUnchanged)
{
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov(0, 1) = 0.05;
    cov(1, 0) = 0.05;
    cov(3, 4) = -0.02;
    cov(4, 3) = -0.02;

    Eigen::Matrix<double, 6, 6> cov_in = cov;
    const double eps = 1e-6;
    double minEig = 0.0;
    ASSERT_TRUE(lio_sam::cov::projectCovarianceToSPD(&cov, eps, &minEig));
    EXPECT_GT(minEig, eps);
    EXPECT_TRUE((cov.isApprox(cov_in, 1e-12)));
}

TEST(CovarianceUtils, ProjectCovariance3ToSPD)
{
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
    cov(1, 1) = -2.0;

    const double eps = 1e-6;
    double minEig = 0.0;
    ASSERT_TRUE(lio_sam::cov::projectCovarianceToSPD(&cov, eps, &minEig));
    EXPECT_LT(minEig, eps);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), eps * 0.999);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
