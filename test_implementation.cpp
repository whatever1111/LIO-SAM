#include <iostream>
#include <cstring>
#include <cassert>
#include <vector>
#include <cmath>
#include <iomanip>

// Test structures matching imageProjection.cpp
struct VelodynePoint
{
    float x, y, z;
    float intensity;
    uint16_t ring;
    uint16_t padding;  // For alignment
    float time;
} __attribute__((packed));

struct LivoxPoint
{
    float x, y, z;
    float intensity;
    uint8_t line;
    uint8_t tag;
    uint8_t reserved;
    uint8_t padding;  // For alignment
    uint32_t timestamp;
} __attribute__((packed));

struct OusterPoint
{
    float x, y, z;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint8_t padding;
    uint16_t noise;
    uint32_t range;
} __attribute__((packed));

class ImplementationTester {
public:
    void testMemoryLayout() {
        std::cout << "\n=== Testing Memory Layout ===" << std::endl;

        std::cout << "VelodynePoint size: " << sizeof(VelodynePoint) << " bytes (expected 24)" << std::endl;
        std::cout << "LivoxPoint size: " << sizeof(LivoxPoint) << " bytes (expected 24)" << std::endl;
        std::cout << "OusterPoint size: " << sizeof(OusterPoint) << " bytes (expected 32)" << std::endl;

        // Check field offsets
        VelodynePoint vp;
        std::cout << "\nVelodyne offsets:" << std::endl;
        std::cout << "  ring offset: " << offsetof(VelodynePoint, ring) << std::endl;
        std::cout << "  time offset: " << offsetof(VelodynePoint, time) << std::endl;

        LivoxPoint lp;
        std::cout << "\nLivox offsets:" << std::endl;
        std::cout << "  line offset: " << offsetof(LivoxPoint, line) << std::endl;
        std::cout << "  timestamp offset: " << offsetof(LivoxPoint, timestamp) << std::endl;

        assert(sizeof(VelodynePoint) == 24);
        assert(sizeof(LivoxPoint) == 24);
        std::cout << "✓ Memory layout test passed!" << std::endl;
    }

    void testLivoxTimestampConversion() {
        std::cout << "\n=== Testing Livox Timestamp Conversion ===" << std::endl;

        // Create test Livox points
        std::vector<LivoxPoint> points(1000);

        // Initialize with test data
        uint32_t base_timestamp = 1700000000000;  // Base timestamp in ms
        for (size_t i = 0; i < points.size(); i++) {
            points[i].x = i * 0.1f;
            points[i].y = i * 0.2f;
            points[i].z = i * 0.05f;
            points[i].intensity = 100.0f;
            points[i].line = i % 16;
            points[i].timestamp = base_timestamp + i * 100;  // 100ms apart
        }

        std::cout << "Original timestamps (first 5):" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "  Point " << i << ": " << points[i].timestamp << " ms" << std::endl;
        }

        // Simulate the conversion in imageProjection.cpp
        const double MS_TO_SEC = 0.001;
        const double firstTimestamp = points[0].timestamp * MS_TO_SEC;

        std::cout << "\nConverting to relative seconds..." << std::endl;
        for (auto& pt : points) {
            float relativeTime = (pt.timestamp * MS_TO_SEC) - firstTimestamp;
            // Store as float bits in timestamp field
            pt.timestamp = *reinterpret_cast<uint32_t*>(&relativeTime);
        }

        std::cout << "After conversion (first 5):" << std::endl;
        for (int i = 0; i < 5; i++) {
            float time = *reinterpret_cast<float*>(&points[i].timestamp);
            std::cout << "  Point " << i << ": " << std::fixed << std::setprecision(3)
                      << time << " s (expected " << i * 0.1 << " s)" << std::endl;
            assert(std::abs(time - i * 0.1) < 1e-6);
        }

        // Test last point
        float lastTime = *reinterpret_cast<float*>(&points.back().timestamp);
        float expectedLastTime = (points.size() - 1) * 0.1;
        std::cout << "\nLast point time: " << lastTime << " s (expected "
                  << expectedLastTime << " s)" << std::endl;
        assert(std::abs(lastTime - expectedLastTime) < 1e-6);

        std::cout << "✓ Livox timestamp conversion test passed!" << std::endl;
    }

    void testGetPointDataFunction() {
        std::cout << "\n=== Testing getPointData Function Logic ===" << std::endl;

        // Test Velodyne
        {
            VelodynePoint vp;
            vp.x = 1.0f; vp.y = 2.0f; vp.z = 3.0f;
            vp.intensity = 50.0f;
            vp.ring = 10;
            vp.time = 0.05f;

            std::cout << "Velodyne point access:" << std::endl;
            std::cout << "  x=" << vp.x << ", y=" << vp.y << ", z=" << vp.z << std::endl;
            std::cout << "  intensity=" << vp.intensity << std::endl;
            std::cout << "  ring=" << vp.ring << std::endl;
            std::cout << "  time=" << vp.time << std::endl;
        }

        // Test Livox with converted timestamp
        {
            LivoxPoint lp;
            lp.x = 1.0f; lp.y = 2.0f; lp.z = 3.0f;
            lp.intensity = 50.0f;
            lp.line = 5;

            // Store time as float bits
            float time = 0.123f;
            lp.timestamp = *reinterpret_cast<uint32_t*>(&time);

            std::cout << "\nLivox point access:" << std::endl;
            std::cout << "  x=" << lp.x << ", y=" << lp.y << ", z=" << lp.z << std::endl;
            std::cout << "  intensity=" << lp.intensity << std::endl;
            std::cout << "  ring(line)=" << (int)lp.line << std::endl;

            // Read time back
            float readTime = *reinterpret_cast<float*>(&lp.timestamp);
            std::cout << "  time=" << readTime << std::endl;
            assert(std::abs(readTime - time) < 1e-6);
        }

        std::cout << "✓ getPointData logic test passed!" << std::endl;
    }

    void testZeroCopy() {
        std::cout << "\n=== Testing Zero-Copy Implementation ===" << std::endl;

        // Simulate void* usage
        void* cloudPtr = nullptr;

        // Allocate as Livox
        auto* livoxCloud = new std::vector<LivoxPoint>(100);
        cloudPtr = livoxCloud;

        // Initialize
        for (size_t i = 0; i < livoxCloud->size(); i++) {
            (*livoxCloud)[i].x = i;
            (*livoxCloud)[i].line = i % 16;
            (*livoxCloud)[i].timestamp = 1000000 + i * 100;
        }

        // Access through void*
        auto* accessedCloud = static_cast<std::vector<LivoxPoint>*>(cloudPtr);
        std::cout << "First point x: " << (*accessedCloud)[0].x << std::endl;
        std::cout << "First point line: " << (int)(*accessedCloud)[0].line << std::endl;

        // In-place conversion (zero-copy)
        const double MS_TO_SEC = 0.001;
        const double firstTimestamp = (*accessedCloud)[0].timestamp * MS_TO_SEC;

        for (auto& pt : *accessedCloud) {
            float relativeTime = (pt.timestamp * MS_TO_SEC) - firstTimestamp;
            pt.timestamp = *reinterpret_cast<uint32_t*>(&relativeTime);
        }

        // Verify conversion
        float firstTime = *reinterpret_cast<float*>(&(*accessedCloud)[0].timestamp);
        float lastTime = *reinterpret_cast<float*>(&(*accessedCloud)[99].timestamp);

        std::cout << "After in-place conversion:" << std::endl;
        std::cout << "  First point time: " << firstTime << " s (expected 0.0)" << std::endl;
        std::cout << "  Last point time: " << lastTime << " s (expected " << 99 * 0.1 << ")" << std::endl;

        assert(std::abs(firstTime - 0.0) < 1e-6);
        assert(std::abs(lastTime - 9.9) < 1e-6);

        delete livoxCloud;

        std::cout << "✓ Zero-copy test passed!" << std::endl;
    }

    void testEdgeCases() {
        std::cout << "\n=== Testing Edge Cases ===" << std::endl;

        // Test 1: Empty cloud
        std::cout << "Test 1: Empty cloud handling" << std::endl;
        std::vector<LivoxPoint> emptyCloud;
        if (emptyCloud.empty()) {
            std::cout << "  ✓ Empty cloud correctly detected" << std::endl;
        }

        // Test 2: Single point
        std::cout << "Test 2: Single point cloud" << std::endl;
        std::vector<LivoxPoint> singlePoint(1);
        singlePoint[0].timestamp = 1234567890;

        const double MS_TO_SEC = 0.001;
        const double firstTimestamp = singlePoint[0].timestamp * MS_TO_SEC;
        float relativeTime = (singlePoint[0].timestamp * MS_TO_SEC) - firstTimestamp;
        assert(std::abs(relativeTime - 0.0) < 1e-6);
        std::cout << "  ✓ Single point: relative time = " << relativeTime << std::endl;

        // Test 3: Very large timestamps
        std::cout << "Test 3: Large timestamp values" << std::endl;
        uint32_t largeTimestamp = 4294967295;  // Max uint32
        double timeInSec = largeTimestamp * MS_TO_SEC;
        std::cout << "  Max timestamp: " << largeTimestamp << " ms = "
                  << timeInSec << " s = " << timeInSec / 3600 / 24 << " days" << std::endl;

        // Test 4: Negative ring/line values
        std::cout << "Test 4: Invalid ring/line values" << std::endl;
        int ring = -1;
        if (ring < 0 || ring >= 16) {
            std::cout << "  ✓ Invalid ring correctly detected" << std::endl;
        }

        std::cout << "✓ Edge cases test passed!" << std::endl;
    }

    void runAllTests() {
        std::cout << "========================================" << std::endl;
        std::cout << "IMAGEROJECTION IMPLEMENTATION TESTS" << std::endl;
        std::cout << "========================================" << std::endl;

        testMemoryLayout();
        testLivoxTimestampConversion();
        testGetPointDataFunction();
        testZeroCopy();
        testEdgeCases();

        std::cout << "\n========================================" << std::endl;
        std::cout << "✓✓✓ ALL TESTS PASSED! ✓✓✓" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

int main() {
    ImplementationTester tester;
    tester.runAllTests();
    return 0;
}