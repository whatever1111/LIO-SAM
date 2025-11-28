#!/usr/bin/env python3
"""
深入分析时间戳问题
"""

import rosbag
import numpy as np
import struct

def analyze_timestamp_issue(bag_path):
    """深入分析时间戳单位和转换问题"""

    bag = rosbag.Bag(bag_path)

    print("="*60)
    print("时间戳问题深入分析")
    print("="*60)

    # 读取一条消息详细分析
    for topic, msg, t in bag.read_messages(topics=['/lidar_points']):
        print(f"\nROS Header 时间戳:")
        print(f"  msg.header.stamp: {msg.header.stamp.to_sec():.6f} 秒")
        print(f"  这是: 2025年11月18日的时间戳（正确）")

        # 解析实际的 timestamp 字段
        fields_info = {}
        for field in msg.fields:
            fields_info[field.name] = {
                'offset': field.offset,
                'datatype': field.datatype
            }

        timestamp_offset = fields_info['timestamp']['offset']
        point_step = msg.point_step

        # 获取前10个点的时间戳
        timestamps = []
        for i in range(0, min(10*point_step, len(msg.data)), point_step):
            # double 类型，8字节
            ts_bytes = msg.data[i+timestamp_offset:i+timestamp_offset+8]
            ts = struct.unpack('d', ts_bytes)[0]
            timestamps.append(ts)

        print(f"\n点云内部 timestamp 字段（前5个点）:")
        for i, ts in enumerate(timestamps[:5]):
            print(f"  点 {i}: {ts:.6f}")

        # 分析时间戳单位
        first_ts = timestamps[0]
        last_ts = timestamps[-1]
        diff = last_ts - first_ts

        print(f"\n时间戳分析:")
        print(f"  第一个点: {first_ts:.6f}")
        print(f"  最后一个点: {last_ts:.6f}")
        print(f"  原始差值: {diff:.6f}")

        # 尝试不同的单位假设
        print(f"\n单位假设测试:")

        # 1. 如果是毫秒级 Unix 时间戳
        if first_ts > 1e12:  # 大于 1e12 可能是毫秒级
            unix_sec = first_ts / 1000.0
            print(f"  假设1: 毫秒级Unix时间戳")
            print(f"    转换为秒: {unix_sec:.6f}")
            print(f"    与ROS时间戳差异: {abs(unix_sec - msg.header.stamp.to_sec()):.6f} 秒")

            # 点之间的时间差
            if diff < 100:  # 差值小于100，可能是毫秒差
                print(f"    点间时间差: {diff:.3f} ms = {diff/1000:.6f} 秒")

        # 2. 如果是微秒级
        if first_ts > 1e15:  # 可能是微秒级
            unix_sec = first_ts / 1e6
            print(f"  假设2: 微秒级Unix时间戳")
            print(f"    转换为秒: {unix_sec:.6f}")
            print(f"    与ROS时间戳差异: {abs(unix_sec - msg.header.stamp.to_sec()):.6f} 秒")

        # 3. 如果是纳秒级
        if first_ts > 1e18:
            unix_sec = first_ts / 1e9
            print(f"  假设3: 纳秒级Unix时间戳")
            print(f"    转换为秒: {unix_sec:.6f}")

        print("\n" + "="*60)
        print("问题诊断:")
        print("="*60)

        # 当前 imageProjection.cpp 的处理
        print("\n当前代码逻辑 (imageProjection.cpp:245):")
        print("  dst.time = (src.timestamp - first.timestamp) / 1000.0;")

        converted_time = diff / 1000.0
        print(f"\n  处理结果: {diff:.6f} / 1000 = {converted_time:.6f} 秒")
        print(f"  预期扫描时间: ~0.1 秒 (10Hz)")
        print(f"  差异: {abs(converted_time - 0.1):.6f} 秒")

        # 正确的处理方式
        print("\n正确的处理方式:")

        # 判断最可能的情况
        if first_ts > 1e12 and first_ts < 2e12:
            # 毫秒级 Unix 时间戳
            print("  时间戳格式: 毫秒级 Unix 时间戳")
            print("  点间差值单位: 毫秒")

            if diff < 10:  # 如果差值小于10，说明已经是毫秒
                print(f"\n  ✅ 正确转换: 差值 {diff:.3f} ms 不需要除以 1000")
                print(f"     应该是: dst.time = (src.timestamp - first.timestamp) / 1000000.0;")
                print(f"     或者: dst.time = (src.timestamp - first.timestamp) * 0.001;")
            elif diff > 1000 and diff < 2000:
                print(f"\n  ⚠️ 可能的问题: 差值 {diff:.1f} 可能是微秒")
                print(f"     应该是: dst.time = (src.timestamp - first.timestamp) / 1000000.0;")

        break

    bag.close()

    print("\n" + "="*60)
    print("修复建议")
    print("="*60)

    print("\n原代码 (imageProjection.cpp:245):")
    print("```cpp")
    print("dst.time = (src.timestamp - tmpCustomCloudIn->points[0].timestamp) / 1000.0;")
    print("```")

    print("\n修改为:")
    print("```cpp")
    print("// 时间戳是毫秒级Unix时间戳，差值单位是毫秒的小数部分")
    print("// 转换为秒需要除以 1000000 而不是 1000")
    print("dst.time = (src.timestamp - tmpCustomCloudIn->points[0].timestamp) / 1000000.0;")
    print("```")

    print("\n或者更安全的版本:")
    print("```cpp")
    print("double time_diff = src.timestamp - tmpCustomCloudIn->points[0].timestamp;")
    print("// 检查差值范围来判断单位")
    print("if (time_diff < 10.0) {")
    print("    // 差值小于10，可能已经是毫秒，转换为秒")
    print("    dst.time = time_diff * 0.001;")
    print("} else if (time_diff < 10000.0) {")
    print("    // 差值在10-10000之间，可能是微秒")
    print("    dst.time = time_diff * 0.000001;")
    print("} else {")
    print("    // 差值很大，可能是纳秒")
    print("    dst.time = time_diff * 0.000000001;")
    print("}")
    print("```")

if __name__ == "__main__":
    bag_path = "/root/autodl-tmp/info_fixed.bag"
    analyze_timestamp_issue(bag_path)