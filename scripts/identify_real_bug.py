#!/usr/bin/env python3
"""
Correctly identify the Livox timestamp format
"""

import numpy as np

def analyze_livox_format():
    print("="*80)
    print("LIVOX TIMESTAMP FORMAT IDENTIFICATION")
    print("="*80)

    # Data from bag
    header_time = 1763444615.982532  # seconds
    timestamps = np.array([
        1763444615982.6328125000,
        1763444615982.6376953125,
        1763444615982.6425781250,
        1763444615982.6477050781,
    ])

    print(f"\nGiven data:")
    print(f"  Header timestamp: {header_time:.6f} seconds")
    print(f"  First point timestamp: {timestamps[0]:.10f}")

    print("\n" + "="*60)
    print("HYPOTHESIS: Timestamps are MILLISECONDS with fractional part")
    print("="*60)

    # Convert to seconds
    timestamps_sec = timestamps / 1000.0
    print(f"\nAfter dividing by 1000:")
    for i, ts in enumerate(timestamps_sec[:4]):
        print(f"  Point {i}: {ts:.9f} seconds")

    print(f"\nCompare with header:")
    print(f"  Header: {header_time:.9f} seconds")
    print(f"  Point 0: {timestamps_sec[0]:.9f} seconds")
    print(f"  Difference: {abs(timestamps_sec[0] - header_time):.9f} seconds")

    if abs(timestamps_sec[0] - header_time) < 0.001:
        print(f"  ✓ PERFECT MATCH! Timestamps ARE in milliseconds")

    # Calculate scan duration
    print(f"\nScan duration analysis:")
    duration_ms = timestamps[-1] - timestamps[0]
    duration_sec = duration_ms / 1000.0
    print(f"  In milliseconds: {duration_ms:.6f} ms")
    print(f"  In seconds: {duration_sec:.6f} seconds")

    print("\n" + "="*60)
    print("THE REAL BUG")
    print("="*60)

    print("\nThe Livox timestamps are stored as:")
    print("  • Double values like: 1763444615982.6328")
    print("  • This represents: MILLISECONDS.FRACTIONAL_MS")
    print("  • To convert to seconds: value / 1000.0")

    print("\nOur current code INCORRECTLY does:")
    print("  • MS_TO_SEC = 0.001")
    print("  • timestamp * MS_TO_SEC = value * 0.001 = value / 1000")
    print("  • This IS actually correct!")

    print("\nWait, let's check the relative time calculation:")
    first_ts_ms = timestamps[0]
    first_ts_sec = first_ts_ms / 1000.0

    print(f"\n1. Original approach (our current code):")
    for i, ts in enumerate(timestamps[:4]):
        relative = (ts / 1000.0) - first_ts_sec
        print(f"  Point {i}: {relative:.9f} seconds")

    # But our code had a bug - we were storing back as uint32!
    print("\n2. THE ACTUAL BUG - We were treating double as uint32!")
    print("   In the old code, we had:")
    print("     uint32_t timestamp;  // WRONG TYPE!")
    print("   But data is actually:")
    print("     double timestamp;    // 8 bytes, not 4!")

    print("\n" + "="*60)
    print("ROOT CAUSE FOUND")
    print("="*60)
    print("\n✗ The struct had wrong field type (uint32_t instead of double)")
    print("✗ This caused memory corruption when reading/writing timestamps")
    print("✗ The conversion logic (MS to seconds) is actually correct")
    print("\n✓ Fix: Use 'double timestamp' in LivoxPoint struct")
    print("✓ Conversion: (timestamp_ms / 1000.0) - first_timestamp_sec")

if __name__ == "__main__":
    analyze_livox_format()