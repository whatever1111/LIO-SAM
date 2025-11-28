#include <iostream>
#include <cstring>

int main() {
    // Test Livox timestamp conversion
    uint32_t timestamp = 1234567890;  // Original milliseconds

    // Convert to relative seconds
    const double MS_TO_SEC = 0.001;
    float relativeTime = (timestamp * MS_TO_SEC) - 1234567.0;

    std::cout << "Original timestamp: " << timestamp << " ms" << std::endl;
    std::cout << "Converted to relative seconds: " << relativeTime << " s" << std::endl;

    // Store float bits back in uint32_t
    uint32_t storedAsUint = *reinterpret_cast<uint32_t*>(&relativeTime);
    std::cout << "Stored as uint32_t: " << storedAsUint << std::endl;

    // Read back as float
    float readBack = *reinterpret_cast<float*>(&storedAsUint);
    std::cout << "Read back as float: " << readBack << " s" << std::endl;

    // Verify correctness
    if (readBack == relativeTime) {
        std::cout << "✓ Conversion is correct!" << std::endl;
    } else {
        std::cout << "✗ Conversion failed!" << std::endl;
    }

    // Test boundary case
    std::cout << "\nTesting boundary case:" << std::endl;
    uint32_t firstTimestamp = 1000000000;
    uint32_t lastTimestamp = 1000100000;  // 100 seconds later

    float firstTime = 0.0f;  // First point is always 0
    float lastTime = (lastTimestamp - firstTimestamp) * MS_TO_SEC;

    std::cout << "First point time: " << firstTime << " s" << std::endl;
    std::cout << "Last point time: " << lastTime << " s" << std::endl;
    std::cout << "Scan duration: " << lastTime << " s" << std::endl;

    return 0;
}