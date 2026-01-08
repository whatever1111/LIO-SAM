# Mapping status published by mapOptmization (per /lio_sam/mapping/odometry_incremental message).
#
# Motivation:
# - Upstream LIO-SAM historically encoded a "degenerate" flag into Odometry.pose.covariance[0].
# - This is fragile because covariance fields should represent uncertainty, and downstream tools may
#   treat them as a real covariance matrix.
# - We publish the flag explicitly to keep covariance semantics correct and avoid accidental misuse.
Header header
bool is_degenerate
