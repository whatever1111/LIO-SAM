# GPS and LIO-SAM Covariance Analysis Report

Generated: 2025-12-09 18:00:44

## Data Summary

| Source | Messages | Duration (s) | Rate (Hz) |
|--------|----------|--------------|-----------|
| gps_odometry | 8768 | 880.0 | 10.0 |
| gps_odomenu | 8768 | 880.0 | 10.0 |

## Position Covariance Statistics

| Source | Valid % | Mean Std X (m) | Mean Std Y (m) | Mean Std Z (m) |
|--------|---------|----------------|----------------|----------------|
| gps_odometry | 100.0 | 0.5713 | 0.5550 | 0.6665 |
| gps_odomenu | 100.0 | 0.5524 | 0.5915 | 0.6509 |

## Orientation Covariance Statistics

| Source | Valid % | Mean Std Roll (deg) | Mean Std Pitch (deg) | Mean Std Yaw (deg) |
|--------|---------|---------------------|----------------------|--------------------|
| gps_odometry | 100.0 | 2.45 | 1.75 | 2.47 |
| gps_odomenu | 100.0 | 1.79 | 2.27 | 2.60 |

## Key Findings

- GPS provides **dynamic position covariance** with range [0.0002, 4.2597] m² for X
- Average GPS position uncertainty: ~0.60 m (1σ)

## Recommendations

1. **GPS Covariance**: Use `useGpsSensorCovariance: true` to leverage dynamic GPS uncertainty
2. **LIO-SAM Covariance**: The internal `poseCovariance` from ISAM2 marginals is available but not published
3. To publish LIO-SAM covariance, modify `mapOptmization.cpp:publishOdometry()` to include `poseCovariance`