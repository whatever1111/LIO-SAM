# LIO-SAM Diagnostic Analysis Report

Generated: 2025-12-08 00:45:25

## System Health Score

**Score: 80.0/100**

### Issues Detected:
- High GPS-Fusion 2D error: 207.96m

## Anomaly Statistics

## GPS-Fusion Error (Time-Aligned)

### 3D Error
- Mean: 207.996 m
- Std: 201.751 m
- Max: 632.471 m

### 2D Error (XY plane)
- Mean: 207.959 m
- Std: 201.713 m
- Max: 632.399 m

### End Point Error
- 3D: 632.471 m
- 2D: 632.399 m

## Optimization Suggestions

1. GPS-Fusion mean 2D error (207.96m) is relatively high. Consider adjusting gpsCovThreshold or poseCovThreshold.

2. Consider reducing gpsNoiseScale to increase GPS weight, or check GPS extrinsics configuration.

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
