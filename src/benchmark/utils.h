// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once

#include <sstream>
#include <iostream>
#include <functional>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include "zipf.h"
#include "omp.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <limits>
#include "tbb/parallel_sort.h"

#define PROFILE 1

#if !defined(HELPER_H)
#define HELPER_H

#define COUT_THIS(this) std::cout << this << std::endl;
#define COUT_VAR(this) std::cout << #this << ": " << this << std::endl;
#define COUT_POS() COUT_THIS("at " << __FILE__ << ":" << __LINE__)
#define COUT_N_EXIT(msg) \
  COUT_THIS(msg);        \
  COUT_POS();            \
  abort();
#define INVARIANT(cond)            \
  if (!(cond)) {                   \
    COUT_THIS(#cond << " failed"); \
    COUT_POS();                    \
    abort();                       \
  }

// Debug logging.
// Default: OFF in release builds (when NDEBUG is defined), ON otherwise.
// Compatibility: also honor legacy NDEBUGGING.
#if defined(NDEBUG) || defined(NDEBUGGING)
#define DEBUG_THIS(this)
#else
#define DEBUG_THIS(this) std::cerr << this << std::endl
#endif

#define UNUSED(var) ((void)var)

#define CACHELINE_SIZE (1 << 6)

#define PACKED __attribute__((packed))

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#ifndef FENCE
#define FENCE

inline void memory_fence() { asm volatile("mfence" : : : "memory"); }

#endif
#ifndef MEMORY_FENCE
#define MEMORY_FENCE

/** @brief Compiler fence.
 * Prevents reordering of loads and stores by the compiler. Not intended to
 * synchronize the processor's caches. */
inline void fence() { asm volatile("" : : : "memory"); }

#endif

inline uint64_t cmpxchg(uint64_t *object, uint64_t expected,
                        uint64_t desired) {
    asm volatile("lock; cmpxchgq %2,%1"
    : "+a"(expected), "+m"(*object)
    : "r"(desired)
    : "cc");
    fence();
    return expected;
}

inline uint8_t cmpxchgb(uint8_t *object, uint8_t expected,
                        uint8_t desired) {
    asm volatile("lock; cmpxchgb %2,%1"
    : "+a"(expected), "+m"(*object)
    : "r"(desired)
    : "cc");
    fence();
    return expected;
}

#endif  // HELPER_H

struct System {
    static void profile(const std::string &name, std::function<void()> body) {
        std::string filename = name.find(".data") == std::string::npos ? (name + ".data") : name;

        // Launch profiler
        pid_t pid;
#ifdef PROFILE
        std::stringstream s;
        s << getpid();
#endif
        int ppid = getpid();
        pid = fork();
        if (pid == 0) {
            // perf to generate the record file
#ifdef PROFILE
            auto fd = open("/dev/null", O_RDWR);
            dup2(fd, 1);
            dup2(fd, 2);
            exit(execl("/usr/bin/perf", "perf", "record", "-o", filename.c_str(), "-p", s.str().c_str(), nullptr));
#else
            // perf the cache misses of the file
            char buf[200];
            //sprintf(buf, "perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,r412e -p %d > %s 2>&1",ppid,filename.c_str());
            sprintf(buf, "perf stat -p %d > %s 2>&1",ppid,filename.c_str());
            execl("/bin/sh", "sh", "-c", buf, NULL);
#endif
        }
#ifndef PROFILE
        setpgid(pid, 0);
#endif
        sleep(3);
        // Run body
        body();
        // Kill profiler
#ifdef PROFILE
        kill(pid, SIGINT);
#else
        kill(-pid, SIGINT);
#endif
        sleep(1);
//waitpid(pid,nullptr,0);
    }

    static void profile(std::function<void()> body) {
        profile("perf.data", body);
    }
};

template<class T>
long long load_binary_data(T *&data, long long length, const std::string &file_path) {
    // open key file
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if (!is.is_open()) {
        return 0;
    }

    std::cout << file_path << std::endl;

    // read the number of keys
    T max_size;
    is.read(reinterpret_cast<char*>(&max_size), sizeof(T));

    std::cout << max_size << std::endl;

    // create array
    if(length < 0 || length > max_size) length = max_size;
    data = new T[length];

    // read keys (chunked: avoids potential issues with very large single read requests)
    const size_t total_bytes = static_cast<size_t>(length) * sizeof(T);
    constexpr size_t kChunkBytes = 64ULL << 20; // 64 MiB
    size_t done = 0;
    char *dst = reinterpret_cast<char *>(data);
    while (done < total_bytes) {
        const size_t want = std::min(kChunkBytes, total_bytes - done);
        is.read(dst + done, static_cast<std::streamsize>(want));
        const std::streamsize got = is.gcount();
        if (got != static_cast<std::streamsize>(want)) {
            // Mark failure for caller; leave a helpful hint.
            std::cerr << "[load_binary_data] short read: want=" << want
                      << " got=" << got
                      << " done=" << done
                      << " total=" << total_bytes
                      << " path=" << file_path << "\n";
            is.close();
            delete[] data;
            data = nullptr;
            return 0;
        }
        done += want;
    }
    is.close();
    std::cout << "Loaded " << length << " keys from " << file_path << std::endl;
    return length;
}

template<class T>
long long load_text_data(T *&array, long long length, const std::string &file_path) {
    std::ifstream is(file_path.c_str());
    if (!is.is_open()) {
        return 0;
    }
    long long i = 0;
    std::string str;

    std::vector<T> temp_keys;
    temp_keys.reserve(200000000);
    while (std::getline(is, str) && (i < length || length < 0)) {
        std::istringstream ss(str);
        T key;
        ss >> key;
        temp_keys.push_back(key);
        i++;
    }

    array = new T[temp_keys.size()];
    for(int j = 0; j < temp_keys.size(); j++) {
        array[j] = temp_keys[j];
    }
    is.close();
    return temp_keys.size();
}

// Duplicate check for a dataset file.
// - Loads all keys into memory (assume memory is sufficient).
// - Sorts keys.
// - Scans adjacent values to find duplicates.
//
// Returns true if the check ran successfully; has_duplicates is set accordingly.
template<class T>
struct DupCheckResult {
    bool ok = false;
    bool has_duplicates = false;
    long long n = 0;
    long long dup_adjacent_pairs = 0; // after sorting, count of i where a[i]==a[i-1]
    T first_dup_value{};
};

template<class T>
DupCheckResult<T> check_duplicates_sorted_scan(const std::string &file_path,
                                               const std::string &file_type = "binary",
                                               long long length = -1) {
    DupCheckResult<T> r;

    T *data = nullptr;
    long long n = 0;
    if (file_type == "binary") {
        n = load_binary_data<T>(data, length, file_path);
    } else if (file_type == "text") {
        n = load_text_data<T>(data, length, file_path);
    } else {
        std::cerr << "[dup_check] unsupported file_type=" << file_type
                  << " (expected binary/text)\n";
        r.ok = false;
        return r;
    }

    if (n <= 0 || data == nullptr) {
        std::cerr << "[dup_check] failed to load keys from " << file_path << "\n";
        r.ok = false;
        return r;
    }

    std::sort(data, data + n);

    long long dup_pairs = 0;
    bool has_dup = false;
    T first_dup{};
    for (long long i = 1; i < n; ++i) {
        if (data[i] == data[i - 1]) {
            ++dup_pairs;
            if (!has_dup) {
                has_dup = true;
                first_dup = data[i];
            }
        }
    }

    delete[] data;
    data = nullptr;

    r.ok = true;
    r.n = n;
    r.has_duplicates = has_dup;
    r.dup_adjacent_pairs = dup_pairs;
    r.first_dup_value = first_dup;
    return r;
}

template<class T>
T *get_search_keys(T array[], int num_keys, int num_searches, size_t *seed = nullptr) {
    auto *keys = new T[num_searches];

#pragma omp parallel
    {
        std::mt19937_64 gen(std::random_device{}());
        if (seed) {
            gen.seed(*seed + omp_get_thread_num());
        }
        std::uniform_int_distribution<int> dis(0, num_keys - 1);
#pragma omp for
        for (int i = 0; i < num_searches; i++) {
            int pos = dis(gen);
            keys[i] = array[pos];
        }
    }

    return keys;
}


bool file_exists(const std::string &str) {
    std::ifstream fs(str);
    return fs.is_open();
}

template<class T>
T *get_search_keys_zipf(T array[], int num_keys, int num_searches, size_t *seed = nullptr) {
    auto *keys = new T[num_searches];
    ScrambledZipfianGenerator zipf_gen(num_keys, seed);
    for (int i = 0; i < num_searches; i++) {
        int pos = zipf_gen.nextValue();
        keys[i] = array[pos];
    }
    return keys;
}

// Unscrambled Zipfian generator:
// - Domain: [0, num_keys)
// - Interpretation: smaller values are hotter (0 is most popular)
//
// This is derived from src/benchmark/zipf.h but intentionally does NOT scramble/hash the output,
// so the output can be treated as a rank.
class UnscrambledZipfianGenerator {
public:
    static constexpr double ZETAN = ScrambledZipfianGenerator::ZETAN;
    static constexpr double ZIPFIAN_CONSTANT = ScrambledZipfianGenerator::ZIPFIAN_CONSTANT;

    explicit UnscrambledZipfianGenerator(uint64_t num_keys, size_t seed)
        : num_keys_(std::max<uint64_t>(1, num_keys)), gen_(seed), dis_(0.0, 1.0) {
        zeta2theta_ = zeta(2);
        alpha_ = 1.0 / (1.0 - ZIPFIAN_CONSTANT);
        eta_ = (1 - std::pow(2.0 / static_cast<double>(num_keys_), 1 - ZIPFIAN_CONSTANT)) /
               (1 - zeta2theta_ / ZETAN);
    }

    uint64_t nextValue() {
        const double u = dis_(gen_);
        const double uz = u * ZETAN;
        uint64_t ret;
        if (uz < 1.0) {
            ret = 0;
        } else if (uz < 1.0 + std::pow(0.5, ZIPFIAN_CONSTANT)) {
            ret = 1;
        } else {
            ret = static_cast<uint64_t>(static_cast<double>(num_keys_) *
                                        std::pow(eta_ * u - eta_ + 1.0, alpha_));
        }
        if (ret >= num_keys_) ret = num_keys_ - 1;
        return ret;
    }

private:
    uint64_t num_keys_ = 1;
    double alpha_ = 0.0;
    double eta_ = 0.0;
    double zeta2theta_ = 0.0;
    std::mt19937_64 gen_;
    std::uniform_real_distribution<double> dis_;

    double zeta(long n) const {
        double sum = 0.0;
        for (long i = 0; i < n; i++) {
            sum += 1.0 / std::pow(i + 1.0, ZIPFIAN_CONSTANT);
        }
        return sum;
    }
};


template<typename T>
T *unique_data(T *key1, size_t &size1, T *key2, size_t &size2) {
    size_t ptr1 = 0;
    size_t ptr2 = 0;

    std::sort(key1, key1 + size1);
    size1 = std::unique(key1, key1 + size1) - key1;
    std::sort(key2, key2 + size2);
    size2 = std::unique(key2, key2 + size2) - key2;

    size_t result = 0;

    while (ptr1 < size1 && ptr2 < size2) {
        while (key1[ptr1] < key2[ptr2] && ptr1 < size1) {
            ptr1++;
        }
        if (key1[ptr1] == key2[ptr2]) {
            ptr2++;
            continue;
        }
        key2[result++] = key2[ptr2++];
    }

    while (ptr2 < size2) {
        key2[result++] = key2[ptr2++];
    }

    size2 = result;
    std::random_shuffle(key2, key2 + size2);

    return &key2[result];
}

struct LatencyStats {
    uint64_t mn = 0;
    uint64_t p50 = 0;
    uint64_t p90 = 0;
    uint64_t p99 = 0;
    uint64_t p999 = 0;
    uint64_t p9999 = 0;
    uint64_t mx = 0;
    double avg = 0.0;
    size_t n = 0;
};

inline uint64_t sampling_interval_from_ratio(double ratio) {
    if (ratio >= 1.0) return 1;
    return std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(1.0 / ratio)));
}

inline void maybe_sample_periodic(std::vector<uint64_t> &vec,
                                  uint64_t &cnt,
                                  uint64_t interval,
                                  uint64_t v_ns) {
    ++cnt;
    if (interval == 0) return;
    if ((cnt % interval) == 0) vec.push_back(v_ns);
}

inline LatencyStats summarize_latencies(std::vector<uint64_t> v,
                                       const char *what,
                                       std::ostream &warn_out) {
    LatencyStats s{};
    s.n = v.size();
    if (v.empty()) {
        warn_out << "[streaming] warning: no latency samples for " << what << "\n";
        return s;
    }

    std::sort(v.begin(), v.end());
    auto pick = [&](double q) -> uint64_t {
        if (v.size() == 1) return v[0];
        size_t idx = static_cast<size_t>(q * (v.size() - 1));
        return v[idx];
    };

    s.mn = v.front();
    s.p50 = pick(0.50);
    s.p90 = pick(0.90);
    s.p99 = pick(0.99);
    s.p999 = pick(0.999);
    s.p9999 = pick(0.9999);
    s.mx = v.back();
    long double sum = 0;
    for (auto x : v) sum += x;
    s.avg = static_cast<double>(sum / v.size());

    if (v.size() < 100) {
        warn_out << "[streaming] warning: only " << v.size() << " latency samples for " << what
                 << " (percentiles may be noisy)\n";
    }
    return s;
}

inline void emit_latency_stats_kv(std::ostream &out,
                                  const char *prefix,
                                  const LatencyStats &s) {
    out << prefix << "_min=" << s.mn << ";";
    out << prefix << "_50=" << s.p50 << ";";
    out << prefix << "_90=" << s.p90 << ";";
    out << prefix << "_99=" << s.p99 << ";";
    out << prefix << "_999=" << s.p999 << ";";
    out << prefix << "_9999=" << s.p9999 << ";";
    out << prefix << "_max=" << s.mx << ";";
    out << prefix << "_avg=" << std::fixed << std::setprecision(2) << s.avg << ";";
    out << std::defaultfloat;
}

// Write (op_index, latency_ns) points to a CSV file for warmup/stability inspection.
// Format:
//   # <optional header>
//   op_idx,lat_ns
//   0,123
//   10000,95
inline void write_latency_curve_csv(const std::string &path,
                                    const std::vector<std::pair<uint64_t, uint64_t>> &pts,
                                    const std::string &header_comment = "") {
    std::ofstream out(path, std::ios::trunc);
    if (!out) COUT_N_EXIT("Failed to open latency curve file for write");
    if (!header_comment.empty()) {
        out << "# " << header_comment << "\n";
    }
    out << "op_idx,lat_ns\n";
    for (const auto &p : pts) {
        out << p.first << "," << p.second << "\n";
    }
    out.close();
}

inline std::vector<std::pair<uint64_t, uint64_t>> downsample_points_uniform(
        const std::vector<std::pair<uint64_t, uint64_t>> &pts,
        size_t max_points) {
    if (max_points == 0 || pts.size() <= max_points) return pts;
    if (max_points == 1) {
        return std::vector<std::pair<uint64_t, uint64_t>>{pts.back()};
    }
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(max_points);
    const double step = static_cast<double>(pts.size() - 1) / static_cast<double>(max_points - 1);
    for (size_t i = 0; i < max_points; ++i) {
        size_t idx = static_cast<size_t>(std::llround(step * static_cast<double>(i)));
        if (idx >= pts.size()) idx = pts.size() - 1;
        out.push_back(pts[idx]);
    }
    return out;
}

// Write a simple SVG plot for warmup/stability inspection.
// - x: op_idx
// - y: latency (ns)
// Notes:
// - No external dependencies; open the .svg in any browser.
// - Points are optionally downsampled for manageable file size.
inline void write_latency_curve_svg(const std::string &path,
                                    const std::vector<std::pair<uint64_t, uint64_t>> &pts_in,
                                    const std::string &title = "",
                                    size_t max_points = 4000,
                                    int width = 1200,
                                    int height = 500) {
    auto pts = downsample_points_uniform(pts_in, max_points);
    std::ofstream out(path, std::ios::trunc);
    if (!out) COUT_N_EXIT("Failed to open latency curve svg for write");

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height
        << "\" fill=\"white\"/>\n";

    const int pad_l = 60, pad_r = 20, pad_t = 30, pad_b = 40;
    const int plot_w = std::max(1, width - pad_l - pad_r);
    const int plot_h = std::max(1, height - pad_t - pad_b);

    // Title
    if (!title.empty()) {
        out << "<text x=\"" << pad_l << "\" y=\"" << (pad_t - 10)
            << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#111\">"
            << title << "</text>\n";
    }

    // Axes box
    out << "<rect x=\"" << pad_l << "\" y=\"" << pad_t << "\" width=\"" << plot_w
        << "\" height=\"" << plot_h << "\" fill=\"none\" stroke=\"#333\" stroke-width=\"1\"/>\n";

    if (pts.empty()) {
        out << "<text x=\"" << pad_l << "\" y=\"" << (pad_t + 20)
            << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#b00\">no samples</text>\n";
        out << "</svg>\n";
        out.close();
        return;
    }

    uint64_t x_min = pts.front().first, x_max = pts.front().first;
    uint64_t y_min = pts.front().second, y_max = pts.front().second;
    std::vector<uint64_t> ys;
    ys.reserve(pts.size());
    for (const auto &p : pts) {
        x_min = std::min<uint64_t>(x_min, p.first);
        x_max = std::max<uint64_t>(x_max, p.first);
        y_min = std::min<uint64_t>(y_min, p.second);
        y_max = std::max<uint64_t>(y_max, p.second);
        ys.push_back(p.second);
    }
    // Robust y scaling: use p99 as the visible max to reduce outlier dominance.
    LatencyStats st = summarize_latencies(ys, "warmup_svg", std::cerr);
    uint64_t y_vis_max = st.p99 > 0 ? st.p99 : y_max;
    if (y_vis_max == 0) y_vis_max = 1;

    auto x_to_px = [&](uint64_t x) -> int {
        if (x_max == x_min) return pad_l;
        const double t = (double)(x - x_min) / (double)(x_max - x_min);
        return pad_l + (int)std::llround(t * plot_w);
    };
    auto y_to_px = [&](uint64_t y) -> int {
        const uint64_t yy = std::min<uint64_t>(y, y_vis_max);
        const double t = (double)yy / (double)y_vis_max;
        return pad_t + (int)std::llround((1.0 - t) * plot_h);
    };

    // Axis labels
    out << "<text x=\"" << pad_l << "\" y=\"" << (pad_t + plot_h + 30)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#111\">op_idx</text>\n";
    out << "<text x=\"10\" y=\"" << (pad_t + 15)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#111\">lat_ns (clipped@p99)</text>\n";
    out << "<text x=\"" << (pad_l + plot_w - 220) << "\" y=\"" << (pad_t + plot_h + 30)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#555\">samples="
        << pts_in.size() << ", plotted=" << pts.size() << "</text>\n";

    // Polyline
    out << "<polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"1\" points=\"";
    for (size_t i = 0; i < pts.size(); ++i) {
        const int x = x_to_px(pts[i].first);
        const int y = y_to_px(pts[i].second);
        out << x << "," << y;
        if (i + 1 < pts.size()) out << " ";
    }
    out << "\"/>\n";

    // Simple stats footer
    out << "<text x=\"" << pad_l << "\" y=\"" << (height - 10)
        << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#111\">"
        << "min=" << st.mn << " p50=" << st.p50 << " p90=" << st.p90
        << " p99=" << st.p99 << " max=" << st.mx << " avg=" << std::fixed << std::setprecision(2) << st.avg
        << "</text>\n";

    out << "</svg>\n";
    out.close();
}


