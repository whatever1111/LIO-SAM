# Covariance Analysis Recommendations

## Summary

### FPA CORRIMU (/fixposition/fpa/corrimu)
- **Accelerometer Covariance**: ALL ZERO - sensor does not provide covariance
  - Recommendation: Use fixed noise parameters in params.yaml
- **Gyroscope Covariance**: ALL ZERO - sensor does not provide covariance

### Standard IMU (/imu/data)
- **Accelerometer Covariance**: ALL ZERO
- **Gyroscope Covariance**: ALL ZERO
- **Orientation Covariance**: ALL ZERO

### fpa_odometry
- **Position Covariance**: Dynamic=True
  - Mean: [0.3264, 0.3080, 0.4442] m²
  - Range: [0.0003, 4.5745] m² (X)
  - **Recommendation**: USE sensor covariance (useGpsSensorCovariance: true)

### fpa_odomenu
- **Position Covariance**: Dynamic=True
  - Mean: [0.3051, 0.3499, 0.4236] m²
  - Range: [0.0002, 4.2597] m² (X)
  - **Recommendation**: USE sensor covariance (useGpsSensorCovariance: true)

## GTSAM Dynamic Covariance Compatibility

### Why Dynamic IMU Covariance is NOT Suitable for Current Code:

1. **PreintegratedImuMeasurements Design**:
   - GTSAM's IMU preintegration uses FIXED noise parameters set at construction time
   - `PreintegrationParams` defines `accelerometerCovariance` and `gyroscopeCovariance`
   - These are continuous-time white noise parameters, NOT per-measurement covariances

2. **Preintegration Theory**:
   - IMU preintegration accumulates measurements between keyframes
   - Uncertainty grows via covariance propagation: Σ(t) = F*Σ(t-1)*F' + Q
   - Q is process noise, assumed constant for Gaussian white noise model
   - Changing Q per-measurement breaks the preintegration theory

3. **Current Code Implementation (imuPreintegration.cpp:246-248)**:
   ```cpp
   p->accelerometerCovariance = Matrix33::Identity() * pow(imuAccNoise, 2);
   p->gyroscopeCovariance = Matrix33::Identity() * pow(imuGyrNoise, 2);
   ```
   - Noise is set ONCE at initialization
   - Cannot be changed per-measurement without recreating the integrator

4. **Alternatives for Using Dynamic Covariance**:
   - **Option A**: Use average covariance to tune imuAccNoise/imuGyrNoise
   - **Option B**: Reject high-covariance IMU measurements (outlier rejection)
   - **Option C**: Use CombinedImuFactor with time-varying bias model
   - **Option D**: Scale pose correction noise based on IMU quality
