#include <iostream>
#include <cstring>
#include <cstdint>

// Test the union structure
struct TestPoint
{
    float x, y, z;
    float intensity;

    union {
        uint16_t ring;      // 2 bytes
        struct {
            uint8_t line;   // 1 byte
            uint8_t tag;    // 1 byte
            uint8_t reserved;
            uint8_t padding;
        };
    };

    union {
        float time;         // 4 bytes
        uint32_t timestamp; // 4 bytes
        uint32_t t;         // 4 bytes
    };
};

int main()
{
    TestPoint pt;

    std::cout << "=== Testing Union Memory Layout ===" << std::endl;
    std::cout << "sizeof(TestPoint) = " << sizeof(TestPoint) << " bytes" << std::endl;
    std::cout << "offset of ring = " << offsetof(TestPoint, ring) << std::endl;
    std::cout << "offset of line = " << offsetof(TestPoint, line) << std::endl;
    std::cout << "offset of time = " << offsetof(TestPoint, time) << std::endl;
    std::cout << "offset of timestamp = " << offsetof(TestPoint, timestamp) << std::endl;

    std::cout << "\n=== Test 1: Ring/Line Conversion ===" << std::endl;

    // Test 1: Set line value and check ring
    memset(&pt, 0, sizeof(pt));
    pt.line = 15;  // Livox line number
    std::cout << "Set line = 15" << std::endl;
    std::cout << "Read ring = " << pt.ring << " (Expected: 15 for little-endian)" << std::endl;
    std::cout << "Binary ring = 0x" << std::hex << pt.ring << std::dec << std::endl;

    // Test 2: Set ring value and check line
    memset(&pt, 0, sizeof(pt));
    pt.ring = 0x0F0E;  // Set ring to a 16-bit value
    std::cout << "\nSet ring = 0x0F0E (3854)" << std::endl;
    std::cout << "Read line = " << (int)pt.line << " (Expected: 0x0E = 14)" << std::endl;

    // Test 3: Simulate Livox line to ring conversion
    memset(&pt, 0, sizeof(pt));
    pt.line = 8;
    pt.tag = 0;
    pt.reserved = 0;
    std::cout << "\n=== Test 2: Livox Scenario ===" << std::endl;
    std::cout << "Livox: Set line=8, tag=0, reserved=0" << std::endl;
    std::cout << "Read as ring = " << pt.ring << std::endl;

    // Problem: When we copy line to ring
    pt.ring = pt.line;  // This overwrites both bytes!
    std::cout << "After pt.ring = pt.line:" << std::endl;
    std::cout << "  ring = " << pt.ring << std::endl;
    std::cout << "  line = " << (int)pt.line << std::endl;

    std::cout << "\n=== Test 3: Time/Timestamp Conversion ===" << std::endl;

    // Test timestamp to time conversion
    pt.timestamp = 1234567890;  // milliseconds
    std::cout << "Set timestamp = " << pt.timestamp << " ms" << std::endl;
    std::cout << "Read as float time = " << pt.time << " (garbage!)" << std::endl;

    // Try to convert
    float converted_time = pt.timestamp * 0.001f;
    std::cout << "Correct conversion: " << converted_time << " seconds" << std::endl;

    // Now assign back to time field
    pt.time = converted_time;
    std::cout << "After pt.time = converted_time:" << std::endl;
    std::cout << "  time = " << pt.time << " seconds" << std::endl;
    std::cout << "  timestamp = " << pt.timestamp << " (garbage!)" << std::endl;

    // The problem: time and timestamp share memory, you can't use both!

    std::cout << "\n=== Test 4: Correct Approach ===" << std::endl;

    // Correct: Convert timestamp and store in time field
    uint32_t orig_timestamp = 1234567890;
    float time_seconds = orig_timestamp * 0.001f - 1234567.0f;  // relative time
    pt.time = time_seconds;

    std::cout << "Original timestamp: " << orig_timestamp << " ms" << std::endl;
    std::cout << "Converted to relative seconds: " << pt.time << std::endl;
    std::cout << "timestamp field is now garbage: " << pt.timestamp << std::endl;

    return 0;
}