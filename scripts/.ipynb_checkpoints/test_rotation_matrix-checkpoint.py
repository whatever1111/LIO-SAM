#!/usr/bin/env python3
"""
Test Eigen quaternion conversion behavior.
This simulates what LIO-SAM does with extrinsicRPY.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# The extrinsicRPY matrix from params.yaml (stored row-major)
extRPY_rowmajor = np.array([
    0, -1, 0,
    1,  0, 0,
    0,  0, 1
]).reshape(3, 3)

print("="*60)
print("Testing extrinsicRPY matrix interpretation")
print("="*60)

print("\n1. extrinsicRPY matrix (as loaded from YAML, row-major):")
print(extRPY_rowmajor)

# Check what rotation this represents
r_extRPY = R.from_matrix(extRPY_rowmajor)
euler_extRPY = r_extRPY.as_euler('xyz', degrees=True)
print(f"\n2. extRPY as Euler angles: Roll={euler_extRPY[0]:.1f}, Pitch={euler_extRPY[1]:.1f}, Yaw={euler_extRPY[2]:.1f}")

# extQRPY = extRPY.inverse()
r_extQRPY = r_extRPY.inv()
euler_extQRPY = r_extQRPY.as_euler('xyz', degrees=True)
print(f"\n3. extQRPY (inverse) as Euler: Roll={euler_extQRPY[0]:.1f}, Pitch={euler_extQRPY[1]:.1f}, Yaw={euler_extQRPY[2]:.1f}")

# Test with sample IMU orientation
# /imu/data has yaw ~ 84.2 degrees
test_yaw = 84.2
r_raw = R.from_euler('xyz', [0, 0, test_yaw], degrees=True)
print(f"\n4. Test raw IMU orientation: Yaw = {test_yaw}°")

# Apply transform: q_final = q_raw * extQRPY
r_final = r_raw * r_extQRPY
euler_final = r_final.as_euler('xyz', degrees=True)
print(f"\n5. After q_raw * extQRPY: Roll={euler_final[0]:.1f}, Pitch={euler_final[1]:.1f}, Yaw={euler_final[2]:.1f}")

# Expected: 84.2 - 90 = -5.8 degrees
expected_yaw = test_yaw - 90
while expected_yaw > 180: expected_yaw -= 360
while expected_yaw < -180: expected_yaw += 360
print(f"\n6. Expected yaw (84.2 - 90): {expected_yaw:.1f}°")

if abs(euler_final[2] - expected_yaw) < 1:
    print("\n[OK] Transform is correct: q * Rz(-90) gives expected result")
else:
    diff = euler_final[2] - expected_yaw
    print(f"\n[ERROR] Transform gives wrong result!")
    print(f"   Expected: {expected_yaw:.1f}°")
    print(f"   Got:      {euler_final[2]:.1f}°")
    print(f"   Difference: {diff:.1f}°")

# Also check what -95.8 implies
print("\n" + "="*60)
print("Additional analysis:")
print("="*60)
print(f"If output yaw is -95.8° (as seen in LIO-SAM log):")
print(f"   With raw yaw = 84.2°")
print(f"   Applied rotation = 84.2 - (-95.8) = {84.2 - (-95.8):.1f}° = 180°")
print(f"   This means Rz(180°) was applied instead of Rz(90°)")
