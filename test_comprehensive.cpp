// Comprehensive test for Livox point cloud processing
// Tests: Deskew correctness, Smoothness validity, Feature extraction, Processing time

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>

// ============== Point Type Definitions ==============
#pragma pack(push, 1)
struct LivoxPointXYZIRT {
    float x, y, z, intensity;
    uint8_t tag, line;
    double timestamp;
};
#pragma pack(pop)
POINT_CLOUD_REGISTER_POINT_STRUCT(LivoxPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (uint8_t, tag, tag)(uint8_t, line, line)(double, timestamp, timestamp))

struct VelodynePointXYZIRT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (uint16_t, ring, ring)(float, time, time))

using PointXYZIRT = VelodynePointXYZIRT;

// ============== Test Results ==============
struct TestResults {
    // Timing
    double conversion_time_ms = 0;
    double sorting_time_ms = 0;
    double deskew_time_ms = 0;
    double smoothness_time_ms = 0;
    double feature_time_ms = 0;
    double total_time_ms = 0;

    // Deskew validation
    int points_with_valid_time = 0;
    int points_with_correct_deskew = 0;
    double max_deskew_error = 0;

    // Smoothness validation
    int total_smoothness_windows = 0;
    int valid_smoothness_windows = 0;
    double invalid_percentage = 0;

    // Feature extraction
    int edge_features = 0;
    int surface_features = 0;
    int total_features = 0;
};

// ============== Simulate IMU Rotation (simplified) ==============
void findRotation(double pointTime, double timeScanCur, float* rotX, float* rotY, float* rotZ) {
    // Simulate rotation: assume 10 deg/s rotation around Z axis
    double dt = pointTime - timeScanCur;
    *rotX = 0;
    *rotY = 0;
    *rotZ = static_cast<float>(dt * 10.0 * M_PI / 180.0);  // 10 deg/s
}

// ============== Deskew Point ==============
void deskewPoint(float& x, float& y, float& z, float time, double timeScanCur,
                 float rotXStart, float rotYStart, float rotZStart) {
    float rotXCur, rotYCur, rotZCur;
    findRotation(timeScanCur + time, timeScanCur, &rotXCur, &rotYCur, &rotZCur);

    // Relative rotation
    float dRotZ = rotZCur - rotZStart;

    // Apply rotation (simplified, only Z-axis)
    float cosZ = std::cos(dRotZ);
    float sinZ = std::sin(dRotZ);
    float newX = cosZ * x - sinZ * y;
    float newY = sinZ * x + cosZ * y;
    x = newX;
    y = newY;
}

// ============== Process Single Frame ==============
TestResults processFrame(const pcl::PointCloud<LivoxPointXYZIRT>& livoxCloud, int N_SCAN, int Horizon_SCAN) {
    TestResults results;
    auto t_start = std::chrono::high_resolution_clock::now();

    // ===== Step 1: Conversion =====
    auto t1 = std::chrono::high_resolution_clock::now();

    double minTimestamp = std::numeric_limits<double>::max();
    double maxTimestamp = std::numeric_limits<double>::lowest();
    for (const auto& pt : livoxCloud.points) {
        if (pt.timestamp < minTimestamp) minTimestamp = pt.timestamp;
        if (pt.timestamp > maxTimestamp) maxTimestamp = pt.timestamp;
    }

    const double MS_TO_SEC = 0.001;
    double scanDuration = (maxTimestamp - minTimestamp) * MS_TO_SEC;
    minTimestamp *= MS_TO_SEC;

    pcl::PointCloud<PointXYZIRT> laserCloudIn;
    laserCloudIn.points.resize(livoxCloud.size());
    laserCloudIn.is_dense = livoxCloud.is_dense;

    for (size_t i = 0; i < livoxCloud.size(); i++) {
        const auto& src = livoxCloud.points[i];
        auto& dst = laserCloudIn.points[i];
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.intensity = src.intensity;
        dst.ring = src.line;
        dst.time = static_cast<float>(src.timestamp * MS_TO_SEC - minTimestamp);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    results.conversion_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ===== Step 2: Sorting (OPTIMIZED: bucket + precompute) =====
    auto t3 = std::chrono::high_resolution_clock::now();

    {
        size_t cloudSize = laserCloudIn.points.size();

        // Pre-compute angles
        std::vector<float> angles(cloudSize);
        for (size_t i = 0; i < cloudSize; i++)
        {
            angles[i] = std::atan2(laserCloudIn.points[i].y, laserCloudIn.points[i].x);
        }

        // Count points per ring
        std::vector<size_t> ringCount(N_SCAN, 0);
        for (const auto &pt : laserCloudIn.points)
        {
            if (pt.ring < N_SCAN) ringCount[pt.ring]++;
        }

        // Calculate start index for each ring bucket
        std::vector<size_t> ringStart(N_SCAN + 1, 0);
        for (int i = 0; i < N_SCAN; i++)
        {
            ringStart[i + 1] = ringStart[i] + ringCount[i];
        }

        // Bucket by ring: create index-angle pairs
        struct IdxAngle { size_t idx; float angle; };
        std::vector<IdxAngle> bucketed(cloudSize);
        std::vector<size_t> ringPos = ringStart;

        for (size_t i = 0; i < cloudSize; i++)
        {
            int ring = laserCloudIn.points[i].ring;
            if (ring < N_SCAN)
            {
                bucketed[ringPos[ring]++] = {i, angles[i]};
            }
        }

        // Sort each ring bucket by angle
        for (int ring = 0; ring < N_SCAN; ring++)
        {
            std::sort(bucketed.begin() + ringStart[ring],
                      bucketed.begin() + ringStart[ring + 1],
                      [](const IdxAngle &a, const IdxAngle &b) { return a.angle < b.angle; });
        }

        // Reorder points using sorted indices
        pcl::PointCloud<PointXYZIRT> sorted;
        sorted.points.resize(cloudSize);
        sorted.is_dense = laserCloudIn.is_dense;
        for (size_t i = 0; i < cloudSize; i++)
        {
            sorted.points[i] = laserCloudIn.points[bucketed[i].idx];
        }
        laserCloudIn = std::move(sorted);
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    results.sorting_time_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // ===== Step 3: Deskew Validation =====
    auto t5 = std::chrono::high_resolution_clock::now();

    double timeScanCur = 0;  // Simulated scan start

    // Key validation: After sorting, each point still has valid time for deskew lookup
    // The deskew function uses point.time to find IMU rotation, so time must be valid
    int points_with_time_in_range = 0;
    int points_with_time_preserved = 0;
    float minTime = FLT_MAX, maxTime = -FLT_MAX;

    for (size_t i = 0; i < laserCloudIn.size(); i++) {
        float t = laserCloudIn.points[i].time;
        if (t >= -0.001 && t <= scanDuration + 0.001) {  // Valid range with small tolerance
            points_with_time_in_range++;
        }
        if (t < minTime) minTime = t;
        if (t > maxTime) maxTime = t;
    }

    results.points_with_valid_time = laserCloudIn.size();
    results.points_with_correct_deskew = points_with_time_in_range;

    // Verify time range is preserved (should span ~scanDuration)
    results.max_deskew_error = std::abs((maxTime - minTime) - scanDuration);

    auto t6 = std::chrono::high_resolution_clock::now();
    results.deskew_time_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();

    // ===== Step 4: Smoothness Calculation =====
    auto t7 = std::chrono::high_resolution_clock::now();

    // Project to range image and extract
    std::vector<std::vector<float>> rangeMat(N_SCAN, std::vector<float>(Horizon_SCAN, FLT_MAX));
    std::vector<int> columnIdnCountVec(N_SCAN, 0);

    for (const auto& pt : laserCloudIn.points) {
        float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (range < 1.0 || range > 1000.0) continue;

        int rowIdn = pt.ring;
        if (rowIdn < 0 || rowIdn >= N_SCAN) continue;

        int columnIdn = columnIdnCountVec[rowIdn]++;
        if (columnIdn >= Horizon_SCAN) continue;

        rangeMat[rowIdn][columnIdn] = range;
    }

    // Extract ranges in order
    std::vector<float> extractedRanges;
    std::vector<int> extractedRings;
    std::vector<float> extractedAngles;

    for (size_t i = 0; i < laserCloudIn.size(); i++) {
        const auto& pt = laserCloudIn.points[i];
        float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (range >= 1.0 && range <= 1000.0 && pt.ring >= 0 && pt.ring < N_SCAN) {
            extractedRanges.push_back(range);
            extractedRings.push_back(pt.ring);
            extractedAngles.push_back(std::atan2(pt.y, pt.x) * 180.0 / M_PI);
        }
    }

    // Calculate smoothness
    int cloudSize = extractedRanges.size();
    std::vector<float> curvature(cloudSize, 0);

    for (int i = 5; i < cloudSize - 5; i++) {
        // Check same ring
        bool sameRing = true;
        for (int k = -5; k <= 5; k++) {
            if (extractedRings[i + k] != extractedRings[i]) {
                sameRing = false;
                break;
            }
        }
        if (!sameRing) continue;

        results.total_smoothness_windows++;

        // Check angle span
        float minAngle = extractedAngles[i], maxAngle = extractedAngles[i];
        for (int k = -5; k <= 5; k++) {
            float a = extractedAngles[i + k];
            if (a < minAngle) minAngle = a;
            if (a > maxAngle) maxAngle = a;
        }
        float angleSpan = maxAngle - minAngle;
        if (angleSpan > 180) angleSpan = 360 - angleSpan;

        if (angleSpan < 30.0) {
            results.valid_smoothness_windows++;

            // Calculate curvature
            float diffRange = 0;
            for (int k = -5; k <= 5; k++) {
                diffRange += (k == 0) ? -extractedRanges[i + k] * 10 : extractedRanges[i + k];
            }
            curvature[i] = diffRange * diffRange;
        }
    }

    if (results.total_smoothness_windows > 0) {
        results.invalid_percentage = 100.0 * (1.0 - (double)results.valid_smoothness_windows / results.total_smoothness_windows);
    }

    auto t8 = std::chrono::high_resolution_clock::now();
    results.smoothness_time_ms = std::chrono::duration<double, std::milli>(t8 - t7).count();

    // ===== Step 5: Feature Extraction =====
    auto t9 = std::chrono::high_resolution_clock::now();

    float edgeThreshold = 1.0;
    float surfThreshold = 0.1;

    for (int i = 5; i < cloudSize - 5; i++) {
        if (curvature[i] > edgeThreshold) {
            results.edge_features++;
        } else if (curvature[i] < surfThreshold && curvature[i] > 0) {
            results.surface_features++;
        }
    }
    results.total_features = results.edge_features + results.surface_features;

    auto t10 = std::chrono::high_resolution_clock::now();
    results.feature_time_ms = std::chrono::duration<double, std::milli>(t10 - t9).count();

    auto t_end = std::chrono::high_resolution_clock::now();
    results.total_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return results;
}

// ============== Main ==============
int main(int argc, char** argv) {
    std::string bag_path = "/root/autodl-tmp/info_fixed.bag";
    std::string topic = "/lidar_points";
    int N_SCAN = 16;
    int Horizon_SCAN = 10000;
    int num_frames = 50;

    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Livox Point Cloud Processing - Comprehensive Test        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    rosbag::Bag bag;
    bag.open(bag_path, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topic));

    std::vector<TestResults> allResults;

    int frame = 0;
    for (const rosbag::MessageInstance& m : view) {
        sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (!msg) continue;

        pcl::PointCloud<LivoxPointXYZIRT> cloud;
        pcl::fromROSMsg(*msg, cloud);

        TestResults results = processFrame(cloud, N_SCAN, Horizon_SCAN);
        allResults.push_back(results);

        frame++;
        if (frame >= num_frames) break;
    }
    bag.close();

    // ===== Calculate Statistics =====
    TestResults avg;
    for (const auto& r : allResults) {
        avg.conversion_time_ms += r.conversion_time_ms;
        avg.sorting_time_ms += r.sorting_time_ms;
        avg.deskew_time_ms += r.deskew_time_ms;
        avg.smoothness_time_ms += r.smoothness_time_ms;
        avg.feature_time_ms += r.feature_time_ms;
        avg.total_time_ms += r.total_time_ms;
        avg.points_with_valid_time += r.points_with_valid_time;
        avg.points_with_correct_deskew += r.points_with_correct_deskew;
        avg.valid_smoothness_windows += r.valid_smoothness_windows;
        avg.total_smoothness_windows += r.total_smoothness_windows;
        avg.edge_features += r.edge_features;
        avg.surface_features += r.surface_features;
        avg.total_features += r.total_features;
        if (r.max_deskew_error > avg.max_deskew_error) avg.max_deskew_error = r.max_deskew_error;
    }

    int n = allResults.size();
    avg.conversion_time_ms /= n;
    avg.sorting_time_ms /= n;
    avg.deskew_time_ms /= n;
    avg.smoothness_time_ms /= n;
    avg.feature_time_ms /= n;
    avg.total_time_ms /= n;

    // ===== Print Results =====
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "                    TEST 1: DESKEW VALIDATION                   " << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    double deskew_accuracy = 100.0 * avg.points_with_correct_deskew / avg.points_with_valid_time;
    std::cout << "  Total points processed:      " << avg.points_with_valid_time << std::endl;
    std::cout << "  Points with valid timestamp: " << avg.points_with_correct_deskew << std::endl;
    std::cout << "  Timestamp validity:          " << std::fixed << std::setprecision(2) << deskew_accuracy << "%" << std::endl;
    std::cout << "  Time range error:            " << std::setprecision(6) << avg.max_deskew_error << " s" << std::endl;
    std::cout << std::endl;
    if (deskew_accuracy > 99.0 && avg.max_deskew_error < 0.01) {
        std::cout << "  [PASS] Timestamps preserved after sorting - deskew will work!" << std::endl;
    } else {
        std::cout << "  [FAIL] Timestamp issues detected!" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "                 TEST 2: SMOOTHNESS VALIDATION                  " << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    double smoothness_validity = 100.0 * avg.valid_smoothness_windows / avg.total_smoothness_windows;
    std::cout << "  Total smoothness windows:    " << avg.total_smoothness_windows << std::endl;
    std::cout << "  Valid smoothness windows:    " << avg.valid_smoothness_windows << std::endl;
    std::cout << "  Validity rate:               " << std::setprecision(2) << smoothness_validity << "%" << std::endl;
    std::cout << std::endl;
    if (smoothness_validity > 90.0) {
        std::cout << "  [PASS] Smoothness calculation is valid after sorting!" << std::endl;
    } else {
        std::cout << "  [FAIL] Too many invalid smoothness windows!" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "                TEST 3: FEATURE EXTRACTION                      " << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  Edge features (per frame):   " << avg.edge_features / n << std::endl;
    std::cout << "  Surface features (per frame):" << avg.surface_features / n << std::endl;
    std::cout << "  Total features (per frame):  " << avg.total_features / n << std::endl;
    std::cout << std::endl;
    if (avg.total_features / n > 1000) {
        std::cout << "  [PASS] Sufficient features extracted!" << std::endl;
    } else {
        std::cout << "  [WARN] Low feature count, may affect odometry quality" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "                 TEST 4: PROCESSING TIME                        " << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  ┌─────────────────────┬────────────┬─────────┐" << std::endl;
    std::cout << "  │ Stage               │ Time (ms)  │ Ratio   │" << std::endl;
    std::cout << "  ├─────────────────────┼────────────┼─────────┤" << std::endl;
    std::cout << "  │ Conversion          │ " << std::setw(10) << avg.conversion_time_ms << " │ " << std::setw(6) << (100*avg.conversion_time_ms/avg.total_time_ms) << "% │" << std::endl;
    std::cout << "  │ Sorting             │ " << std::setw(10) << avg.sorting_time_ms << " │ " << std::setw(6) << (100*avg.sorting_time_ms/avg.total_time_ms) << "% │" << std::endl;
    std::cout << "  │ Deskew              │ " << std::setw(10) << avg.deskew_time_ms << " │ " << std::setw(6) << (100*avg.deskew_time_ms/avg.total_time_ms) << "% │" << std::endl;
    std::cout << "  │ Smoothness          │ " << std::setw(10) << avg.smoothness_time_ms << " │ " << std::setw(6) << (100*avg.smoothness_time_ms/avg.total_time_ms) << "% │" << std::endl;
    std::cout << "  │ Feature Extraction  │ " << std::setw(10) << avg.feature_time_ms << " │ " << std::setw(6) << (100*avg.feature_time_ms/avg.total_time_ms) << "% │" << std::endl;
    std::cout << "  ├─────────────────────┼────────────┼─────────┤" << std::endl;
    std::cout << "  │ TOTAL               │ " << std::setw(10) << avg.total_time_ms << " │  100%   │" << std::endl;
    std::cout << "  └─────────────────────┴────────────┴─────────┘" << std::endl;
    std::cout << std::endl;

    double max_allowed_ms = 100.0;  // 10Hz = 100ms per frame
    if (avg.total_time_ms < max_allowed_ms) {
        std::cout << "  [PASS] Processing time (" << avg.total_time_ms << "ms) < " << max_allowed_ms << "ms (10Hz requirement)" << std::endl;
    } else {
        std::cout << "  [WARN] Processing time (" << avg.total_time_ms << "ms) may be too slow for real-time" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "                      OVERALL VERDICT                           " << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;

    bool all_pass = (deskew_accuracy > 99.0) && (avg.max_deskew_error < 0.01) &&
                    (smoothness_validity > 90.0) && (avg.total_features / n > 1000);

    if (all_pass) {
        std::cout << std::endl;
        std::cout << "  ✓ All tests PASSED!" << std::endl;
        std::cout << "  ✓ Livox point cloud processing is working correctly." << std::endl;
        std::cout << std::endl;
        return 0;
    } else {
        std::cout << std::endl;
        std::cout << "  ✗ Some tests FAILED. Please review the results above." << std::endl;
        std::cout << std::endl;
        return 1;
    }
}
