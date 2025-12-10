# GPS vs LiDAR Position Difference Analysis

## Problem Summary

From the trajectory data, there is a **massive coordinate scale difference** between GPS and LiDAR/Fusion outputs.

## Data Comparison

### GPS Trajectory (gps_trajectory.csv)
- **Start**: x=0.0007m, y=-0.00003m, z=0.0002m (near origin)
- **End**: x=-439.6m, y=-2.35m, z=0.80m
- **Total range**: ~440m in X, ~2.4m in Y, ~0.8m in Z
- **Data pattern**: Smooth, continuous progression

### Fusion/LiDAR Trajectory (fusion_trajectory.csv)
- **Start**: x=-0.02m, y=-0.008m, z=0.037m (near origin)
- **End**: x=-491.8m, y=-35.9m, z=-597.9m
- **Shows severe jumps**: Z values range from -580 to -760m!
- **Data pattern**: Highly oscillatory with sudden jumps (100m+)

## Root Cause Identified

### Problem 1: `useGpsElevation: true` causing Z-axis catastrophe

When you set `useGpsElevation: true`, LIO-SAM uses GPS elevation directly. However:

1. **GPS Z values in the CSV are relative** (0.2m to 0.8m range)
2. **Fusion Z values became extremely negative** (-580 to -760m)
3. This indicates the GPS elevation and LiDAR elevation frames are incompatible

The GPS provides elevation relative to some datum (likely WGS84 ellipsoid or EGM96 geoid),
while LiDAR starts from 0 at the first frame. Mixing them without proper offset causes:
- Factor graph trying to reconcile incompatible Z constraints
- Massive oscillations as optimization fails

### Problem 2: XY Position Divergence

Looking at the end positions:
- GPS: x=-439.6m, y=-2.35m
- Fusion: x=-491.8m, y=-35.9m
- **Difference**: dx=52m, dy=33m

This 60+ meter difference suggests:
1. **Different coordinate frames**: GPS is in ENU (East-North-Up), LiDAR may have different orientation
2. **gpsExtrinsicRot may be incorrect**: Current setting is identity matrix (no rotation)
3. **Accumulating drift** from LiDAR before GPS constraints were properly applied

### Problem 3: Severe Oscillations in Fusion Output

The fusion trajectory shows sudden jumps of 100+ meters between consecutive frames:
```
1763445486482646465: x=-486.8, y=-69.2, z=-634.5
1763445486582731962: x=-325.8, y=+12.0, z=-685.8
```
This 161m jump in one frame indicates the optimizer is thrashing between:
- LiDAR odometry constraints (one estimate)
- GPS factor constraints (conflicting estimate)

## Which One Has the Problem?

### GPS Data: Looks VALID
- Smooth trajectory progression
- Reasonable Y deviation (~2.4m) for a mostly straight path
- Z values stable around 0.8m (reasonable ground-level variation)
- X progresses linearly (~440m total distance)

### Fusion Data: Has PROBLEMS
- Massive Z values (-580 to -760m) are clearly wrong
- Sudden 100m+ jumps between frames
- Y deviation (36m) is much larger than GPS suggests
- Signs of optimizer instability

## Conclusion

**The GPS data appears correct. The problem is in how LIO-SAM processes and fuses it.**

Specific issues:
1. `useGpsElevation: true` is incompatible with your sensor setup
2. GPS constraints are conflicting with LiDAR odometry due to coordinate frame mismatch
3. The optimizer cannot find a consistent solution, causing oscillations

## Recommended Fixes

### Immediate Fix (to stop oscillations):

```yaml
# In params.yaml:
useGpsElevation: false    # REVERT THIS - was causing Z problems
```

### Coordinate Frame Fix:

The gpsExtrinsicRot needs to be calibrated. If GPS is in ENU and LiDAR frame is different:

Current (identity - no rotation):
```yaml
gpsExtrinsicRot: [ 1, 0, 0,
                   0, 1, 0,
                   0, 0, 1]
```

If your LiDAR -X is forward but GPS +East is forward, you may need rotation.

### GPS Weight Tuning:

If GPS and LiDAR disagree by 30-60m, GPS noise should be set higher initially:
```yaml
gpsNoiseMin: 5.0           # Increased from 0.2
gpsNoiseScale: 1.0         # Increased from 0.2
gpsCovThreshold: 10.0      # Be more selective about GPS quality
poseCovThreshold: 25.0     # Re-enable this check (uncomment the code)
```

### Gradual GPS Integration:

Instead of forcing GPS immediately, let the system build confidence:
```cpp
// In addGPSFactor(), increase initial travel distance requirement
if (travel_dist < 10.0)  // Was 3.0, increase to 10m
    return;
```
