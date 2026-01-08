#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace lio_sam
{
namespace cov
{

// Small utilities for handling 3x3 / 6x6 covariances (GNSS, twist, etc.).
//
// Key design goals:
//   - Be explicit about conventions (ROS vs. GTSAM ordering).
//   - Preserve symmetry numerically.
//   - Protect GTSAM/ISAM2 from invalid covariances (NaN/Inf or not SPD).

// Replace non-finite entries (NaN/Inf) with zero and symmetrize.
// Returns true if any entry was changed.
inline bool sanitizeCovariance(Eigen::Matrix<double, 6, 6> *cov)
{
    if (!cov)
        return false;
    bool changed = false;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            const double v = (*cov)(i, j);
            if (!std::isfinite(v))
            {
                (*cov)(i, j) = 0.0;
                changed = true;
            }
        }
    }
    // NOTE: use .eval() to avoid Eigen aliasing issues when the LHS appears on the RHS (transpose reads the same data).
    *cov = (0.5 * (*cov + cov->transpose())).eval();
    return changed;
}

inline bool sanitizeCovariance(Eigen::Matrix3d *cov)
{
    if (!cov)
        return false;
    bool changed = false;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const double v = (*cov)(i, j);
            if (!std::isfinite(v))
            {
                (*cov)(i, j) = 0.0;
                changed = true;
            }
        }
    }
    // NOTE: use .eval() to avoid Eigen aliasing issues when the LHS appears on the RHS.
    *cov = (0.5 * (*cov + cov->transpose())).eval();
    return changed;
}

inline Eigen::Matrix<double, 6, 6> rotateCov6(const Eigen::Matrix<double, 6, 6> &cov_in,
                                              const Eigen::Matrix3d &R_parent_child)
{
    // Rotation of a 6x6 covariance where the layout is [pos(3), rot(3)] and rot is a local so(3) vector.
    // For small-angle rotation vectors, the same 3x3 rotation applies to both translation and rotation blocks.
    // cov_out = T * cov_in * Tᵀ, where T = diag(R, R).
    Eigen::Matrix<double, 6, 6> T = Eigen::Matrix<double, 6, 6>::Zero();
    T.block<3, 3>(0, 0) = R_parent_child;
    T.block<3, 3>(3, 3) = R_parent_child;
    Eigen::Matrix<double, 6, 6> cov_out = T * cov_in * T.transpose();
    return 0.5 * (cov_out + cov_out.transpose()); // enforce symmetry
}

// Reorder pose covariance from ROS convention to GTSAM Pose3 tangent convention.
// ROS nav_msgs/Odometry.pose.covariance order: [pos(x,y,z), rot(x,y,z)] (often documented as roll,pitch,yaw).
// GTSAM Pose3 tangent order: [rot(x,y,z), pos(x,y,z)].
inline Eigen::Matrix<double, 6, 6> reorderRosPoseCovToGtsam(const Eigen::Matrix<double, 6, 6> &cov_ros)
{
    // This is only a permutation of coordinates; it does NOT rotate covariance.
    // The cross terms are also permuted (no transpose mistakes).
    Eigen::Matrix<double, 6, 6> cov_gtsam;
    cov_gtsam.setZero();
    // ori
    cov_gtsam.block<3, 3>(0, 0) = cov_ros.block<3, 3>(3, 3);
    // ori-pos
    cov_gtsam.block<3, 3>(0, 3) = cov_ros.block<3, 3>(3, 0);
    // pos-ori
    cov_gtsam.block<3, 3>(3, 0) = cov_ros.block<3, 3>(0, 3);
    // pos
    cov_gtsam.block<3, 3>(3, 3) = cov_ros.block<3, 3>(0, 0);
    return 0.5 * (cov_gtsam + cov_gtsam.transpose());
}

// Build a Pose3-tangent covariance in GTSAM order [rot, pos] that only constrains orientation.
//
// This is used to fuse GNSS yaw/attitude without letting it distort translation:
// - Translation is made very weak (huge variances, no rot-trans correlations).
// - Optional "yaw-only": roll/pitch are made very weak and yaw is kept from input covariance.
inline Eigen::Matrix<double, 6, 6> makeOrientationOnlyPoseCovGtsam(const Eigen::Matrix<double, 6, 6> &cov_ros,
                                                                   bool yaw_only,
                                                                   double weak_var_roll_pitch,
                                                                   double huge_pos_var)
{
    Eigen::Matrix<double, 6, 6> cov_gtsam = reorderRosPoseCovToGtsam(cov_ros);

    // Zero rotation-translation coupling and disable translation constraints.
    cov_gtsam.block<3, 3>(0, 3).setZero();
    cov_gtsam.block<3, 3>(3, 0).setZero();
    cov_gtsam.block<3, 3>(3, 3).setZero();
    cov_gtsam(3, 3) = huge_pos_var;
    cov_gtsam(4, 4) = huge_pos_var;
    cov_gtsam(5, 5) = huge_pos_var;

    if (yaw_only)
    {
        // Keep only yaw (rot-z) meaningful; roll/pitch are made weak and de-correlated.
        cov_gtsam.row(0).setZero();
        cov_gtsam.col(0).setZero();
        cov_gtsam(0, 0) = weak_var_roll_pitch;
        cov_gtsam.row(1).setZero();
        cov_gtsam.col(1).setZero();
        cov_gtsam(1, 1) = weak_var_roll_pitch;

        // Keep yaw independent from roll/pitch for stability.
        cov_gtsam(0, 2) = 0.0;
        cov_gtsam(2, 0) = 0.0;
        cov_gtsam(1, 2) = 0.0;
        cov_gtsam(2, 1) = 0.0;
    }

    return (0.5 * (cov_gtsam + cov_gtsam.transpose())).eval();
}

// Projects a symmetric covariance matrix to SPD by clamping eigenvalues.
// Returns false only if eigen decomposition fails.
inline bool projectCovarianceToSPD(Eigen::Matrix<double, 6, 6> *cov, double eps, double *min_eig_out = nullptr)
{
    if (!cov)
        return false;

    // Ensure solver sees a symmetric matrix (numerical drift or upstream formatting can break symmetry).
    // NOTE: use .eval() to avoid Eigen aliasing issues.
    *cov = (0.5 * (*cov + cov->transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(*cov);
    if (solver.info() != Eigen::Success)
        return false;

    const double minEig = solver.eigenvalues().minCoeff();
    if (min_eig_out)
        *min_eig_out = minEig;

    if (minEig >= eps)
        return true;

    // Clamp eigenvalues: this is a minimal, deterministic way to make covariance SPD.
    const Eigen::Matrix<double, 6, 1> eval_clamped = solver.eigenvalues().cwiseMax(eps);
    *cov = solver.eigenvectors() * eval_clamped.asDiagonal() * solver.eigenvectors().transpose();
    // Re-symmetrize after reconstruction (again, .eval() avoids aliasing).
    *cov = (0.5 * (*cov + cov->transpose())).eval();
    return true;
}

inline bool projectCovarianceToSPD(Eigen::Matrix3d *cov, double eps, double *min_eig_out = nullptr)
{
    if (!cov)
        return false;

    // Symmetrize before eigendecomposition (and avoid Eigen aliasing).
    *cov = (0.5 * (*cov + cov->transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(*cov);
    if (solver.info() != Eigen::Success)
        return false;

    const double minEig = solver.eigenvalues().minCoeff();
    if (min_eig_out)
        *min_eig_out = minEig;

    if (minEig >= eps)
        return true;

    // Clamp eigenvalues to enforce SPD.
    const Eigen::Vector3d eval_clamped = solver.eigenvalues().cwiseMax(eps);
    *cov = solver.eigenvectors() * eval_clamped.asDiagonal() * solver.eigenvectors().transpose();
    *cov = (0.5 * (*cov + cov->transpose())).eval();
    return true;
}

} // namespace cov
} // namespace lio_sam
