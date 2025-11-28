// performance_timer.h - 性能计时工具
#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <ros/ros.h>

class PerformanceTimer {
private:
    struct TimerStats {
        double total_time = 0;
        double max_time = 0;
        double min_time = 1e9;
        int count = 0;
        std::vector<double> recent_times;

        void addTime(double time_ms) {
            total_time += time_ms;
            max_time = std::max(max_time, time_ms);
            min_time = std::min(min_time, time_ms);
            count++;

            recent_times.push_back(time_ms);
            if (recent_times.size() > 100) {
                recent_times.erase(recent_times.begin());
            }
        }

        double getAverage() const {
            return count > 0 ? total_time / count : 0;
        }

        double getP95() const {
            if (recent_times.empty()) return 0;
            std::vector<double> sorted = recent_times;
            std::sort(sorted.begin(), sorted.end());
            int idx = sorted.size() * 0.95;
            return sorted[idx];
        }
    };

    static std::unordered_map<std::string, TimerStats> timers;
    static std::mutex mutex;
    static ros::Time last_print_time;

public:
    class ScopedTimer {
    private:
        std::string name;
        std::chrono::high_resolution_clock::time_point start;
        bool stopped = false;

    public:
        ScopedTimer(const std::string& timer_name) : name(timer_name) {
            start = std::chrono::high_resolution_clock::now();
        }

        ~ScopedTimer() {
            if (!stopped) {
                stop();
            }
        }

        void stop() {
            if (stopped) return;

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

            std::lock_guard<std::mutex> lock(mutex);
            timers[name].addTime(elapsed_ms);
            stopped = true;
        }
    };

    static void printStats(bool force = false) {
        ros::Time now = ros::Time::now();
        if (!force && (now - last_print_time).toSec() < 5.0) {
            return;  // 每5秒打印一次
        }
        last_print_time = now;

        std::lock_guard<std::mutex> lock(mutex);

        std::cout << "\n========== LIO-SAM Performance Report ==========" << std::endl;
        std::cout << std::setw(30) << "Module"
                  << std::setw(10) << "Avg(ms)"
                  << std::setw(10) << "Min(ms)"
                  << std::setw(10) << "Max(ms)"
                  << std::setw(10) << "P95(ms)"
                  << std::setw(8) << "Count" << std::endl;
        std::cout << std::string(78, '-') << std::endl;

        // 按平均时间排序
        std::vector<std::pair<std::string, TimerStats>> sorted_timers(timers.begin(), timers.end());
        std::sort(sorted_timers.begin(), sorted_timers.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.getAverage() > b.second.getAverage();
                  });

        double total_time = 0;
        for (const auto& [name, stats] : sorted_timers) {
            std::cout << std::setw(30) << name
                     << std::setw(10) << std::fixed << std::setprecision(2) << stats.getAverage()
                     << std::setw(10) << stats.min_time
                     << std::setw(10) << stats.max_time
                     << std::setw(10) << stats.getP95()
                     << std::setw(8) << stats.count << std::endl;

            if (name.find("TOTAL") != std::string::npos) {
                total_time = stats.getAverage();
            }
        }

        // 警告信息
        if (total_time > 200) {
            std::cout << "\n⚠️ WARNING: Average total time " << total_time << "ms exceeds 200ms!" << std::endl;
        }

        // 找出瓶颈
        if (!sorted_timers.empty()) {
            std::cout << "\n🔴 Top bottlenecks:" << std::endl;
            for (int i = 0; i < std::min(3, (int)sorted_timers.size()); i++) {
                const auto& [name, stats] = sorted_timers[i];
                if (name.find("TOTAL") == std::string::npos) {
                    double percentage = total_time > 0 ? (stats.getAverage() / total_time * 100) : 0;
                    std::cout << "  • " << name << ": " << stats.getAverage()
                             << "ms (" << percentage << "% of total)" << std::endl;
                }
            }
        }

        std::cout << "================================================\n" << std::endl;
    }

    static void reset() {
        std::lock_guard<std::mutex> lock(mutex);
        timers.clear();
    }
};

// 静态成员初始化
std::unordered_map<std::string, PerformanceTimer::TimerStats> PerformanceTimer::timers;
std::mutex PerformanceTimer::mutex;
ros::Time PerformanceTimer::last_print_time = ros::Time(0);

// 使用宏简化计时
#define PERF_TIMER(name) PerformanceTimer::ScopedTimer _timer_##__LINE__(name)
#define PERF_TIMER_START(name) auto _timer_##name = std::chrono::high_resolution_clock::now()
#define PERF_TIMER_END(name) do { \
    auto _end_##name = std::chrono::high_resolution_clock::now(); \
    double _elapsed_##name = std::chrono::duration<double, std::milli>(_end_##name - _timer_##name).count(); \
    PerformanceTimer::timers[#name].addTime(_elapsed_##name); \
} while(0)