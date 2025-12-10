# LIO-SAM Diagnostic Analysis Report

Generated: 2025-12-07 21:02:04

## System Health Score

**Score: 100/100**

## Anomaly Statistics

## GPS-Fusion Error (Time-Aligned)

### 3D Error
- Mean: 1.416 m
- Std: 2.515 m
- Max: 15.041 m

### 2D Error (XY plane)
- Mean: 1.337 m
- Std: 2.546 m
- Max: 15.039 m

### End Point Error
- 3D: 1.059 m
- 2D: 0.345 m

## Optimization Suggestions

1. IMU yaw noise (0.64 deg) is high. Consider checking gyroscope calibration or increasing imuGyrNoise.

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
