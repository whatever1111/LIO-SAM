# LIO-SAM Diagnostic Analysis Report

Generated: 2025-12-07 18:56:57

## System Health Score

**Score: 80.0/100**

### Issues Detected:
- High GPS-Fusion error: 10.85m

## Anomaly Statistics

## GPS-Fusion Error

- Mean: 10.851 m
- Std: 8.791 m
- Max: 22.011 m

## Optimization Suggestions

1. GPS-Fusion mean error (10.85m) is high. Consider adjusting gpsCovThreshold or poseCovThreshold.

2. IMU yaw drift (-59.61 deg/min) detected. Consider checking gyroscope bias or GPS heading correction.

3. Consider reducing gpsNoiseScale to increase GPS weight, or check GPS extrinsics configuration.

## Parameter Tuning Guide


### GPS Weight Control (params.yaml)
- `gpsCovThreshold`: Increase to accept more GPS data (current: 5.0)
- `poseCovThreshold`: Increase to add GPS factors more frequently (current: 25.0)
- `gpsNoiseMin`: Decrease for higher GPS weight (current: 0.5)
- `gpsNoiseScale`: Decrease for higher GPS weight (current: 0.5)
- `gpsAddInterval`: Decrease for more frequent GPS factors (current: 3.0)

### IMU Parameters (params.yaml)
- `imuAccNoise`: Accelerometer white noise
- `imuGyrNoise`: Gyroscope white noise
- `imuAccBiasN`: Accelerometer bias random walk
- `imuGyrBiasN`: Gyroscope bias random walk
- `imuGravity`: Local gravity magnitude

### Key Covariance Thresholds
- GPS Factor is added when: `poseCovariance(3,3) >= poseCovThreshold` OR `poseCovariance(4,4) >= poseCovThreshold`
- GPS data is rejected when: `noise_x > gpsCovThreshold` OR `noise_y > gpsCovThreshold`
