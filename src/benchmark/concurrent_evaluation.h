#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../tscns.h"

#include "./flags.h"
#include "./utils.h"

#include "omp.h"

#include "../competitor/competitor.h"

namespace ce {

template<typename KEY_TYPE, typename PAYLOAD_TYPE>
class ConcurrentStreamingReplayBenchmark {
  using index_t = indexInterface<KEY_TYPE, PAYLOAD_TYPE>;

  struct SpinBarrier {
    explicit SpinBarrier(size_t parties_) : parties(parties_) {}
    void wait() {
      const size_t g = gen.load(std::memory_order_acquire);
      if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == parties) {
        count.store(0, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
      } else {
        while (gen.load(std::memory_order_acquire) == g) {
          std::this_thread::yield();
        }
      }
    }
    std::atomic<size_t> count{0};
    std::atomic<size_t> gen{0};
    const size_t parties;
  };

  struct HeartbeatSpinBarrier {
    explicit HeartbeatSpinBarrier(size_t parties_) : parties(parties_) {}
    template<typename HeartbeatFn, typename OnLastFn>
    void wait(HeartbeatFn &&heartbeat_fn, OnLastFn &&on_last_fn) {
      const size_t g = gen.load(std::memory_order_acquire);
      if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == parties) {
        on_last_fn();
        count.store(0, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
      } else {
        uint32_t spin = 0;
        while (gen.load(std::memory_order_acquire) == g) {
          if (((++spin) & 0xFFu) == 0) {
            heartbeat_fn();
          } else {
            std::this_thread::yield();
          }
        }
      }
    }
    template<typename HeartbeatFn>
    void wait(HeartbeatFn &&heartbeat_fn) {
      wait(std::forward<HeartbeatFn>(heartbeat_fn), []() {});
    }
    std::atomic<size_t> count{0};
    std::atomic<size_t> gen{0};
    const size_t parties;
  };

  struct DatasetPair {
    std::string keys_file;
    std::string ts_file;
  };

  struct QueryOp {
    uint64_t ts = 0;
    KEY_TYPE key{};
    KEY_TYPE end_key{};
  };
  struct InsertOp {
    uint64_t ts = 0;
    KEY_TYPE key{};
  };
  struct ReplayOp {
    uint64_t ts = 0;
    KEY_TYPE key{};
    KEY_TYPE aux{};
    uint8_t type = 0;
  };

  struct Precomputed {
    uint64_t n = 0;
    uint64_t time_window = 0;
    uint64_t init_end = 0;
    std::vector<InsertOp> insert_ops;
    std::vector<QueryOp> query_ops;
    std::vector<ReplayOp> ops;
    uint64_t end_ts = 0;
    uint64_t del_ops_n = 0;
    uint64_t ins_ops_n = 0;
    uint64_t query_ops_n = 0;
  };

  std::vector<DatasetPair> dataset_pairs_;
  std::vector<std::string> all_index_type;
  std::string index_type;

  std::string keys_file_path;
  std::string ts_file_path;

  std::string keys_file_type = "binary";
  std::string ts_file_type = "binary";
  long long table_size = -1; // informational only in replay mode
  size_t time_window = 1000000;
  std::string stream_mode = "lookup"; // lookup | scan
  size_t scan_num = 100;
  size_t random_seed = 1;
  double rw_ratio = 1.0;
  double query_hit_ratio = 1.0;
  std::string query_distribution = "uniform"; // uniform | zipf
  std::string zipf_generator = "unscrambled"; // unscrambled | scrambled
  bool shuffle_keys = false;
  bool half_range = false;
  bool stronger_non_duplicate = false;

  std::string output_path = "./out_streaming.txt";
  std::string pre_run_path = ""; // if empty: derived from output_path + signature
  std::vector<std::string> all_pre_run_path_;
  std::string segment_file_path = "";

  size_t threads = 1;
  size_t round_ops = 100000;
  std::string dispatch = "hash"; // hash | round_robin | random

  size_t warm_up = 0;
  bool latency_sample = false;
  double latency_sample_ratio = 0.01;

  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> init_kv_sorted_;
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> init_kv_arrival_;

public:
  static std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string val;
    while (std::getline(iss, val, ',')) out.push_back(val);
    if (out.empty()) out.push_back("");
    return out;
  }

  static uint64_t fnv1a64(const std::string &s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
      h ^= static_cast<uint64_t>(c);
      h *= 1099511628211ULL;
    }
    return h;
  }

  static std::string hex_u64(uint64_t x) {
    static const char *kHex = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
      out[static_cast<size_t>(i)] = kHex[x & 0xFULL];
      x >>= 4ULL;
    }
    return out;
  }

  void parse_args(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);

    const std::string keys_csv = get_required(flags, "keys_file");
    const std::vector<std::string> keys_files = split_csv(keys_csv);
    std::vector<std::string> ts_files;
    const auto it_t = flags.find("ts_file");
    if (it_t == flags.end() || it_t->second.empty()) {
      ts_files.clear();
    } else {
      ts_files = split_csv(it_t->second);
    }
    if (ts_files.size() > keys_files.size()) {
      COUT_N_EXIT("ts_file list longer than keys_file list (expected <=, or omit for generated)");
    }
    while (ts_files.size() < keys_files.size()) ts_files.push_back("");

    dataset_pairs_.clear();
    dataset_pairs_.reserve(keys_files.size());
    for (size_t i = 0; i < keys_files.size(); ++i) {
      dataset_pairs_.push_back(DatasetPair{keys_files[i], ts_files[i]});
    }

    keys_file_path = dataset_pairs_[0].keys_file;
    ts_file_path = dataset_pairs_[0].ts_file;

    keys_file_type = get_with_default(flags, "keys_file_type", "binary");
    ts_file_type = get_with_default(flags, "ts_file_type", keys_file_type);
    table_size = std::stoll(get_with_default(flags, "table_size", "-1"));
    time_window = static_cast<size_t>(std::stoull(get_with_default(flags, "time_window", "1000000")));
    stream_mode = get_with_default(flags, "stream_mode", "lookup");
    scan_num = static_cast<size_t>(std::stoull(get_with_default(flags, "scan_num", "100")));
    random_seed = static_cast<size_t>(std::stoull(get_with_default(flags, "seed", "1")));
    rw_ratio = std::stod(get_with_default(flags, "rw_ratio", "1.0"));
    query_hit_ratio = std::stod(get_with_default(flags, "query_hit_ratio", "1.0"));
    query_distribution = get_with_default(flags, "query_distribution", "uniform");
    zipf_generator = get_with_default(flags, "zipf_generator", "unscrambled");
    shuffle_keys = get_boolean_flag(flags, "shuffle_keys");
    half_range = get_boolean_flag(flags, "half_range");
    stronger_non_duplicate =
        (flags.find("stronger_non_duplicate") != flags.end()) || (flags.find("stronger_non-duplicate") != flags.end());

    output_path = get_with_default(flags, "output_path", "./out_streaming.txt");
    all_pre_run_path_ = split_csv(get_with_default(flags, "pre_run_path", ""));
    pre_run_path = all_pre_run_path_.empty() ? std::string("") : all_pre_run_path_[0];
    segment_file_path = get_with_default(flags, "segment_file", "");

    threads = static_cast<size_t>(std::stoull(get_with_default(flags, "threads", "1")));
    round_ops = static_cast<size_t>(std::stoull(get_with_default(flags, "round_ops", "100000")));
    dispatch = get_with_default(flags, "dispatch", "hash");

    warm_up = static_cast<size_t>(std::stoull(get_with_default(flags, "warm_up", "0")));
    latency_sample = get_boolean_flag(flags, "latency_sample");
    latency_sample_ratio = std::stod(get_with_default(flags, "latency_sample_ratio", "0.01"));

    all_index_type = get_comma_separated(flags, "index");
    if (all_index_type.empty()) {
      COUT_N_EXIT("Required flag --index was not found (e.g., --index=xindex,btreeolc)");
    }
    auto normalize = [](const std::string &s) -> std::string {
      if (s == "alexolc") return "alexol";
      if (s == "lippolc") return "lippol";
      return s;
    };
    for (auto &s : all_index_type) s = normalize(s);
    index_type = all_index_type[0];

    INVARIANT(time_window > 0);
    INVARIANT(threads > 0);
    INVARIANT(round_ops > 0);
    INVARIANT(dispatch == "hash" || dispatch == "round_robin" || dispatch == "random");
    INVARIANT(stream_mode == "lookup" || stream_mode == "scan");
    INVARIANT(scan_num > 0);
    INVARIANT(rw_ratio >= 0.0);
    INVARIANT(query_hit_ratio >= 0.0 && query_hit_ratio <= 1.0);
    INVARIANT(query_distribution == "uniform" || query_distribution == "zipf");
    if (query_distribution == "zipf") {
      INVARIANT(zipf_generator == "unscrambled" || zipf_generator == "scrambled");
    }
    if (latency_sample) {
      INVARIANT(latency_sample_ratio > 0.0);
      INVARIANT(latency_sample_ratio <= 1.0);
    }

    for (const auto &s : all_index_type) {
      if (!(s == "xindex" || s == "finedex" || s == "alexol" || s == "lippol" ||
            s == "sali" || s == "btreeolc" || s == "artolc" || s == "pswix")) {
        std::ostringstream oss;
        oss << "concurrent_evaluation: unsupported --index=" << s
            << " (allowed: xindex,finedex,alexol(alexolc),lippol(lippolc),sali,btreeolc,artolc,pswix)";
        COUT_N_EXIT(oss.str().c_str());
      }
    }

    if (get_boolean_flag(flags, "pre_run")) {
      COUT_N_EXIT("concurrent_evaluation: replay-only; generate pre_run.bin using benchmark_streaming.h --pre_run");
    }
    if (get_boolean_flag(flags, "warmup_check")) {
      COUT_N_EXIT("concurrent_evaluation: warmup_check not supported (use benchmark_streaming.h)");
    }
  }

  void run() {
    for (size_t di = 0; di < dataset_pairs_.size(); ++di) {
      keys_file_path = dataset_pairs_[di].keys_file;
      ts_file_path = dataset_pairs_[di].ts_file;

      const bool pre_path_zipped = (!all_pre_run_path_.empty() && all_pre_run_path_.size() == dataset_pairs_.size());
      const std::vector<std::string> pre_paths =
          pre_path_zipped ? std::vector<std::string>{all_pre_run_path_[di]} : all_pre_run_path_;
      const std::vector<std::string> eff_pre_paths = pre_paths.empty() ? std::vector<std::string>{""} : pre_paths;

      for (const auto &prp : eff_pre_paths) {
        pre_run_path = prp;

        Precomputed pre_init = read_pre_run_file(pre_run_file_path());
        INVARIANT(pre_init.time_window == static_cast<uint64_t>(time_window));

        const size_t init_end = static_cast<size_t>(pre_init.init_end);
        init_kv_sorted_.clear();
        init_kv_sorted_.reserve(init_end);
        init_kv_arrival_.clear();
        init_kv_arrival_.reserve(init_end);
        for (size_t i = 0; i < init_end; ++i) {
          const auto kv =
              std::make_pair(pre_init.insert_ops[i].key, static_cast<PAYLOAD_TYPE>(pre_init.insert_ops[i].ts));
          init_kv_arrival_.push_back(kv);
          init_kv_sorted_.push_back(kv);
        }
        std::sort(init_kv_sorted_.begin(), init_kv_sorted_.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        const std::string out_path = result_file_path_for_config();
        for (const auto &idx_type : all_index_type) {
          index_type = idx_type;
          Precomputed pre = read_pre_run_file(pre_run_file_path());
          INVARIANT(pre.time_window == static_cast<uint64_t>(time_window));
          build_replay_ops(pre);
          std::vector<InsertOp>().swap(pre.insert_ops);
          std::vector<QueryOp>().swap(pre.query_ops);
          run_one_index(pre, out_path);
        }
        init_kv_sorted_.clear();
        init_kv_arrival_.clear();
      }
    }
  }

private:

  std::string result_file_path_for_config() const {
    std::ostringstream sig;
    sig << keys_file_path << "|"
        << ts_file_path << "|"
        << keys_file_type << "|"
        << ts_file_type << "|"
        << "tw=" << time_window << "|"
        << "sm=" << stream_mode << "|"
        << "scan=" << scan_num << "|"
        << "seed=" << random_seed << "|"
        << "rw=" << std::setprecision(17) << rw_ratio << "|"
        << "hr=" << std::setprecision(17) << query_hit_ratio << "|"
        << "qd=" << query_distribution << "|"
        << "zg=" << zipf_generator << "|"
        << "shuf=" << (shuffle_keys ? 1 : 0) << "|"
        << "half=" << (half_range ? 1 : 0) << "|"
        << "snd=" << (stronger_non_duplicate ? 1 : 0) << "|"
        << "pre=" << pre_run_path << "|"
        << "ls=" << (latency_sample ? 1 : 0) << "|"
        << "lsr=" << std::setprecision(17) << latency_sample_ratio << "|"
        << "thr=" << threads << "|"
        << "round=" << round_ops << "|"
        << "disp=" << dispatch;
    const uint64_t h = fnv1a64(sig.str());
    return output_path + ".cfg-" + hex_u64(h) + ".txt";
  }

  std::string config_kv_string() const {
    std::ostringstream out;
    out << "KeysFile=" << keys_file_path << ";";
    out << "TsFile=" << (ts_file_path.empty() ? std::string("(generated)") : ts_file_path) << ";";
    out << "KeysFileType=" << keys_file_type << ";";
    out << "TsFileType=" << ts_file_type << ";";
    out << "StreamMode=" << stream_mode << ";";
    out << "ScanNum=" << scan_num << ";";
    out << "Seed=" << random_seed << ";";
    out << "RWRatio=" << std::setprecision(17) << rw_ratio << ";";
    out << "QueryHitRatio=" << std::setprecision(17) << query_hit_ratio << ";";
    out << "QueryDistribution=" << query_distribution << ";";
    out << "ZipfGenerator=" << zipf_generator << ";";
    out << "PreRunPath=" << pre_run_path << ";";
    out << "LatencySample=" << (latency_sample ? 1 : 0) << ";";
    out << "LatencySampleRatio=" << std::setprecision(17) << latency_sample_ratio << ";";
    out << "WarmUp=" << warm_up << ";";
    out << "Threads=" << threads << ";";
    out << "RoundOps=" << round_ops << ";";
    out << "Dispatch=" << dispatch << ";";
    out << "ShuffleKeys=" << (shuffle_keys ? 1 : 0) << ";";
    out << "HalfRange=" << (half_range ? 1 : 0) << ";";
    out << "StrongerNonDuplicate=" << (stronger_non_duplicate ? 1 : 0) << ";";
    out << std::defaultfloat;
    return out.str();
  }

  static bool path_looks_like_dir(const std::string &p) {
    if (p.empty()) return false;
    const char last = p.back();
    return last == '/' || last == '\\';
  }

  static std::vector<uint64_t> read_u64_list_one_line(const std::string &path) {
    std::vector<uint64_t> v;
    if (path.empty()) return v;
    std::ifstream in(path);
    if (!in) {
      COUT_N_EXIT("Failed to open segment_file");
    }
    std::string line;
    if (!std::getline(in, line)) return v;
    std::istringstream iss(line);
    uint64_t x;
    while (iss >> x) v.push_back(x);
    in.close();
    for (size_t i = 1; i < v.size(); ++i) {
      INVARIANT(v[i] >= v[i - 1]);
    }
    return v;
  }

  uint64_t pre_run_sig_hash() const {
    std::ostringstream sig;
    sig << keys_file_path << "|"
        << ts_file_path << "|"
        << keys_file_type << "|"
        << ts_file_type << "|"
        << "tw=" << time_window << "|"
        << "sm=" << stream_mode << "|"
        << "scan=" << scan_num << "|"
        << "seed=" << random_seed << "|"
        << "rw=" << std::setprecision(17) << rw_ratio << "|"
        << "hr=" << std::setprecision(17) << query_hit_ratio << "|"
        << "qd=" << query_distribution << "|"
        << "zg=" << zipf_generator << "|"
        << "shuf=" << (shuffle_keys ? 1 : 0) << "|"
        << "half=" << (half_range ? 1 : 0) << "|"
        << "snd=" << (stronger_non_duplicate ? 1 : 0);
    return fnv1a64(sig.str());
  }

  uint64_t pre_run_sig_hash_legacy_() const {
    std::ostringstream sig;
    sig << keys_file_path << "|"
        << ts_file_path << "|"
        << keys_file_type << "|"
        << ts_file_type << "|"
        << "tw=" << time_window << "|"
        << "sm=" << stream_mode << "|"
        << "scan=" << scan_num << "|"
        << "seed=" << random_seed << "|"
        << "rw=" << std::setprecision(17) << rw_ratio << "|"
        << "hr=" << std::setprecision(17) << query_hit_ratio << "|"
        << "qd=" << query_distribution << "|"
        << "zg=" << zipf_generator << "|"
        << "shuf=" << (shuffle_keys ? 1 : 0) << "|"
        << "half=" << (half_range ? 1 : 0) << "|"
        << "snd=" << (stronger_non_duplicate ? 1 : 0) << "|"
        << "ls=" << (latency_sample ? 1 : 0) << "|"
        << "lsr=" << std::setprecision(17) << latency_sample_ratio;
    return fnv1a64(sig.str());
  }

  static std::filesystem::path derived_pre_run_dir_path_with_hash_(const std::string &out, uint64_t h) {
    return std::filesystem::path(out + ".pre_run." + hex_u64(h));
  }

  std::filesystem::path derived_pre_run_dir_path() const {
    return derived_pre_run_dir_path_with_hash_(output_path, pre_run_sig_hash());
  }

  std::filesystem::path pre_run_dir_path() const {
    if (pre_run_path.empty()) return derived_pre_run_dir_path();
    if (path_looks_like_dir(pre_run_path)) return std::filesystem::path(pre_run_path);
    return std::filesystem::path(pre_run_path).parent_path();
  }

  std::filesystem::path pre_run_bin_path() const {
    if (pre_run_path.empty()) {
      const auto p = pre_run_dir_path() / "pre_run.bin";
      if (!std::filesystem::exists(p)) {
        const auto legacy_dir = derived_pre_run_dir_path_with_hash_(output_path, pre_run_sig_hash_legacy_());
        const auto legacy = legacy_dir / "pre_run.bin";
        if (std::filesystem::exists(legacy)) return legacy;
      }
      return p;
    }
    if (path_looks_like_dir(pre_run_path)) return pre_run_dir_path() / "pre_run.bin";
    return std::filesystem::path(pre_run_path);
  }

  std::string pre_run_file_path() const { return pre_run_bin_path().string(); }

  static constexpr uint64_t kPreMagic = 0x4752455F50524538ULL;

  Precomputed read_pre_run_file(const std::string &path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      COUT_N_EXIT("Failed to open pre_run_path for read (run benchmark_streaming.h --pre_run first)");
    }

    uint64_t magic = 0, n = 0, tw = 0, init_end = 0;
    uint64_t scan_num_u64 = 0, seed_u64 = 0, flags = 0;
    uint64_t insert_n = 0, query_n = 0;
    double rw_ratio_f64 = 0.0;
    double hit_ratio_f64 = 0.0;
    double latency_ratio_f64 = 0.0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&tw), sizeof(tw));
    in.read(reinterpret_cast<char*>(&init_end), sizeof(init_end));
    INVARIANT(magic == kPreMagic);
    in.read(reinterpret_cast<char*>(&scan_num_u64), sizeof(scan_num_u64));
    in.read(reinterpret_cast<char*>(&seed_u64), sizeof(seed_u64));
    in.read(reinterpret_cast<char*>(&rw_ratio_f64), sizeof(rw_ratio_f64));
    in.read(reinterpret_cast<char*>(&hit_ratio_f64), sizeof(hit_ratio_f64));
    in.read(reinterpret_cast<char*>(&latency_ratio_f64), sizeof(latency_ratio_f64));
    in.read(reinterpret_cast<char*>(&flags), sizeof(flags));
    in.read(reinterpret_cast<char*>(&insert_n), sizeof(insert_n));
    in.read(reinterpret_cast<char*>(&query_n), sizeof(query_n));

    INVARIANT(scan_num_u64 == static_cast<uint64_t>(scan_num));
    INVARIANT(seed_u64 == static_cast<uint64_t>(random_seed));
    INVARIANT(rw_ratio_f64 == rw_ratio);
    INVARIANT(hit_ratio_f64 == query_hit_ratio);
    (void)latency_ratio_f64;

    const uint64_t cur_flags =
        (stream_mode == "scan" ? 1ULL : 0ULL) |
        (query_distribution == "zipf" ? 2ULL : 0ULL) |
        (zipf_generator == "scrambled" ? 4ULL : 0ULL) |
        (shuffle_keys ? 8ULL : 0ULL);
    INVARIANT(flags == cur_flags);

    Precomputed p;
    p.n = n;
    p.time_window = tw;
    p.init_end = init_end;
    INVARIANT(insert_n == n);
    p.insert_ops.resize(static_cast<size_t>(insert_n));
    p.query_ops.resize(static_cast<size_t>(query_n));
    for (size_t i = 0; i < static_cast<size_t>(insert_n); ++i) {
      InsertOp ins;
      in.read(reinterpret_cast<char*>(&ins.key), sizeof(KEY_TYPE));
      in.read(reinterpret_cast<char*>(&ins.ts), sizeof(ins.ts));
      p.insert_ops[i] = ins;
    }
    for (size_t i = 0; i < static_cast<size_t>(query_n); ++i) {
      QueryOp q;
      in.read(reinterpret_cast<char*>(&q.key), sizeof(KEY_TYPE));
      in.read(reinterpret_cast<char*>(&q.end_key), sizeof(KEY_TYPE));
      in.read(reinterpret_cast<char*>(&q.ts), sizeof(q.ts));
      p.query_ops[i] = q;
    }
    in.close();
    return p;
  }

  static void build_replay_ops(Precomputed &p) {
    p.ops.clear();
    p.del_ops_n = 0;
    p.ins_ops_n = 0;
    p.query_ops_n = 0;
    p.end_ts = !p.insert_ops.empty() ? p.insert_ops.back().ts : 0ULL;

    const uint64_t w = p.time_window;
    const size_t init_end = static_cast<size_t>(p.init_end);
    const size_t n = static_cast<size_t>(p.n);

    uint64_t del_n_u64 = 0;
    if (!p.insert_ops.empty() && p.end_ts > w + 1ULL) {
      const uint64_t thr = p.end_ts - w - 1ULL;
      del_n_u64 = static_cast<uint64_t>(
          std::upper_bound(p.insert_ops.begin(), p.insert_ops.end(), thr,
                           [](uint64_t x, const InsertOp &op) { return x < op.ts; }) -
          p.insert_ops.begin());
    }
    const size_t del_n = static_cast<size_t>(del_n_u64);
    const size_t ins_n = (n > init_end) ? (n - init_end) : 0;
    const size_t q_n = p.query_ops.size();

    p.del_ops_n = del_n_u64;
    p.ins_ops_n = static_cast<uint64_t>(ins_n);
    p.query_ops_n = static_cast<uint64_t>(q_n);

    p.ops.reserve(del_n + ins_n + q_n);

    for (size_t i = 0; i < del_n; ++i) {
      const uint64_t base = p.insert_ops[i].ts + w;
      const uint64_t ev = (base == std::numeric_limits<uint64_t>::max()) ? base : (base + 1ULL);
      if (ev > p.end_ts) break;
      ReplayOp op;
      op.ts = ev;
      op.key = p.insert_ops[i].key;
      op.aux = static_cast<KEY_TYPE>(0);
      op.type = 0; // DEL
      p.ops.push_back(op);
    }
    for (const auto &q : p.query_ops) {
      ReplayOp op;
      op.ts = q.ts;
      op.key = q.key;
      op.aux = q.end_key;
      op.type = 1; // QUERY
      p.ops.push_back(op);
    }
    for (size_t i = init_end; i < n; ++i) {
      ReplayOp op;
      op.ts = p.insert_ops[i].ts;
      op.key = p.insert_ops[i].key;
      op.aux = static_cast<KEY_TYPE>(0);
      op.type = 2; // INS
      p.ops.push_back(op);
    }

    std::sort(p.ops.begin(), p.ops.end(),
              [](const ReplayOp &a, const ReplayOp &b) {
                if (a.ts != b.ts) return a.ts < b.ts;
                return a.type < b.type; // DEL -> QUERY -> INS
              });
  }

  index_t* prepare_index() {
    Param param(threads, 0);
    index_t *index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);
    index->init(&param);
    if (index_type == "pswix") {
      index->bulk_load(init_kv_arrival_.data(), init_kv_arrival_.size(), &param);
    } else {
      index->bulk_load(init_kv_sorted_.data(), init_kv_sorted_.size(), &param);
    }
    return index;
  }

  void run_one_index(Precomputed &pre, const std::string &out_path) {
    const bool is_pswix = (index_type == "pswix");
    if (is_pswix) {
      pswix::configure_runtime_params(threads, static_cast<uint64_t>(time_window),
                                      round_ops);
    }

    std::cerr << "[concurrent_eval] index=" << index_type
              << " phase=bulk_load begin"
              << " init_kv=" << init_kv_sorted_.size()
              << " threads=" << threads
              << std::endl;
    index_t *index = prepare_index();

    std::cerr << "[concurrent_eval] index=" << index_type
              << " phase=bulk_load end"
              << std::endl;

    const uint64_t nops_u64 = static_cast<uint64_t>(pre.ops.size());
    const uint64_t warm_u64 = static_cast<uint64_t>(std::min(warm_up, pre.ops.size()));
    const uint64_t warm_full_rounds =
        (round_ops == 0) ? 0 : (warm_u64 / static_cast<uint64_t>(round_ops));
    const uint64_t warm_barriers = warm_full_rounds + 1ULL;
    const uint64_t rem = (nops_u64 > warm_u64) ? (nops_u64 - warm_u64) : 0ULL;
    const uint64_t meas_full_rounds =
        (round_ops == 0) ? 0 : (rem / static_cast<uint64_t>(round_ops));
    const uint64_t meas_barriers = meas_full_rounds + 1ULL;
    const uint64_t total_barriers = warm_barriers + meas_barriers;

    std::vector<uint8_t> sample_mask;
    if (latency_sample && is_pswix) {
      sample_mask.resize(static_cast<size_t>(nops_u64), 0);
      std::ostringstream ss;
      ss << "latency_sample_mask|"
         << "sig=" << hex_u64(pre_run_sig_hash()) << "|"
         << "seed=" << random_seed << "|"
         << "ratio=" << std::setprecision(17) << latency_sample_ratio;
      std::mt19937_64 rng(fnv1a64(ss.str()));
      std::bernoulli_distribution bern(latency_sample_ratio);
      for (size_t i = 0; i < static_cast<size_t>(nops_u64); ++i) {
        sample_mask[i] = static_cast<uint8_t>(bern(rng) ? 1 : 0);
      }
    }

    static constexpr size_t kCacheLine = 64;
    struct alignas(kCacheLine) WorkerStat {
      uint64_t sample_cycles = 0;
      uint64_t sample_ops = 0;
      uint64_t scan_tuples = 0;
      uint8_t _pad[kCacheLine - 3 * sizeof(uint64_t)]{};
    };
    static_assert(sizeof(WorkerStat) == kCacheLine, "WorkerStat must occupy exactly one cache line");
    std::vector<WorkerStat> stats(threads);

    std::vector<std::vector<ReplayOp>> per_tid_ops;
    std::vector<std::vector<uint32_t>> per_tid_round_counts_warm;
    std::vector<std::vector<uint32_t>> per_tid_round_counts_meas;
    std::vector<size_t> per_tid_warm_end;
    std::vector<std::vector<uint64_t>> per_tid_sample_bits;

    if (!is_pswix) {
      per_tid_ops.assign(threads, {});
      per_tid_round_counts_warm.assign(threads, std::vector<uint32_t>(static_cast<size_t>(warm_barriers), 0));
      per_tid_round_counts_meas.assign(threads, std::vector<uint32_t>(static_cast<size_t>(meas_barriers), 0));
      per_tid_warm_end.assign(threads, 0);
      per_tid_sample_bits.assign(threads, {});

      for (auto &v : per_tid_ops) v.reserve(static_cast<size_t>(nops_u64 / threads + 64));

      std::mt19937_64 rng_dispatch;
      std::uniform_int_distribution<size_t> uni_tid(0, threads - 1);
      if (dispatch == "random") {
        std::ostringstream ss;
        ss << "dispatch=random|"
           << "sig=" << hex_u64(pre_run_sig_hash()) << "|"
           << "seed=" << random_seed << "|"
           << "thr=" << threads;
        rng_dispatch.seed(fnv1a64(ss.str()));
      }

      std::mt19937_64 rng_sample;
      std::bernoulli_distribution bern_sample(latency_sample_ratio);
      if (latency_sample) {
        std::ostringstream ss;
        ss << "latency_sample_mask|"
           << "sig=" << hex_u64(pre_run_sig_hash()) << "|"
           << "seed=" << random_seed << "|"
           << "ratio=" << std::setprecision(17) << latency_sample_ratio;
        rng_sample.seed(fnv1a64(ss.str()));
      }

      auto set_sample_bit = [&](size_t tid, size_t op_idx) {
        auto &bits = per_tid_sample_bits[tid];
        const size_t w = op_idx >> 6;
        if (bits.size() <= w) bits.resize(w + 1, 0ULL);
        bits[w] |= (1ULL << (op_idx & 63ULL));
      };

      for (uint64_t pos = 0; pos < nops_u64; ++pos) {
        const ReplayOp &rop = pre.ops[static_cast<size_t>(pos)];

        size_t tid = 0;
        if (threads == 1) {
          tid = 0;
        } else if (dispatch == "random") {
          tid = uni_tid(rng_dispatch);
        } else if (dispatch == "round_robin") {
          tid = static_cast<size_t>(pos % static_cast<uint64_t>(threads));
        } else { // hash
          tid = (std::hash<KEY_TYPE>{}(rop.key) % threads);
        }

        const size_t op_idx = per_tid_ops[tid].size();
        per_tid_ops[tid].push_back(rop);

        if (pos < warm_u64) {
          const size_t r =
              (round_ops == 0) ? 0 : static_cast<size_t>(pos / static_cast<uint64_t>(round_ops));
          per_tid_round_counts_warm[tid][r] += 1;
          per_tid_warm_end[tid] += 1;
        } else {
          const uint64_t rel = pos - warm_u64;
          const size_t r =
              (round_ops == 0) ? 0 : static_cast<size_t>(rel / static_cast<uint64_t>(round_ops));
          per_tid_round_counts_meas[tid][r] += 1;
        }

        if (latency_sample) {
          const bool sample = bern_sample(rng_sample);
          if (sample && pos >= warm_u64) set_sample_bit(tid, op_idx);
        }
      }

      std::vector<ReplayOp>().swap(pre.ops);
    }

    auto exec_one = [&](Param &param,
                        PAYLOAD_TYPE &tmp,
                        std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> &scan_buf,
                        const ReplayOp &rop) -> uint64_t {
      param.ts = rop.ts;
      if (rop.type == 0) { // DEL
        (void)index->remove(static_cast<KEY_TYPE>(rop.key), &param);
        return 0;
      } else if (rop.type == 1) { // QUERY
        if (stream_mode == "lookup") {
          (void)index->get(static_cast<KEY_TYPE>(rop.key), tmp, &param);
          return 0;
        } else {
          param.scan_end_key = static_cast<uint64_t>(rop.aux);
          return static_cast<uint64_t>(
              index->scan(static_cast<KEY_TYPE>(rop.key), scan_num, scan_buf.data(), &param));
        }
      } else { // INS
        (void)index->put(static_cast<KEY_TYPE>(rop.key),
                         static_cast<PAYLOAD_TYPE>(rop.ts),
                         &param);
        return 0;
      }
    };

    INVARIANT(threads > 0);
    INVARIANT(threads <= static_cast<size_t>(std::numeric_limits<int>::max()));

    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_end;
    double wall_meas_s = 0.0;
    const bool heartbeat_wait = (index_type == "xindex");
    const KEY_TYPE heartbeat_key =
        init_kv_sorted_.empty() ? KEY_TYPE{} : static_cast<KEY_TYPE>(init_kv_sorted_.front().first);
    std::atomic<size_t> warm_done_cnt{0};
    std::atomic<size_t> meas_done_cnt{0};
    std::atomic<bool> measured_go{false};
    std::atomic<bool> measured_end_recorded{false};

    const bool seg_enabled = !segment_file_path.empty();
    const std::vector<uint64_t> seg_round_ends_raw = read_u64_list_one_line(segment_file_path);
    const uint64_t max_round_end_u64 = static_cast<uint64_t>(meas_barriers);
    std::vector<uint64_t> seg_round_ends;
    seg_round_ends.reserve(seg_round_ends_raw.size());
    for (const uint64_t x : seg_round_ends_raw) {
      if (x >= 1ULL && x <= max_round_end_u64) seg_round_ends.push_back(x);
    }
    struct SegmentRec {
      size_t seg_idx = 0;
      uint64_t round_start = 0; // inclusive (0-based measured round index)
      uint64_t round_end = 0;   // exclusive (0-based)
      uint64_t ops = 0;         // total executed ops in this segment across all threads
      double wall_end_s = 0.0;
    };
    std::vector<SegmentRec> segment_recs;
    if (seg_enabled) segment_recs.reserve(std::max<size_t>(1, seg_round_ends.size() + 1));
    std::unique_ptr<HeartbeatSpinBarrier> hb_round_barrier;
    if (seg_enabled && !seg_round_ends.empty() && heartbeat_wait) {
      hb_round_barrier = std::make_unique<HeartbeatSpinBarrier>(threads);
    }

    std::vector<uint64_t> meas_round_ops; // size == meas_barriers
    std::vector<uint64_t> meas_round_ops_prefix; // prefix sum for fast segment ops calculation
    if (seg_enabled) {
      meas_round_ops.assign(static_cast<size_t>(meas_barriers), 0ULL);
      if (is_pswix) {
        const uint64_t rem_u64 = (nops_u64 > warm_u64) ? (nops_u64 - warm_u64) : 0ULL;
        const uint64_t meas_tail_u64 =
            (round_ops == 0) ? 0ULL : (rem_u64 % static_cast<uint64_t>(round_ops));
        for (size_t r = 0; r < meas_round_ops.size(); ++r) {
          uint64_t c = (r + 1 == meas_round_ops.size())
                           ? meas_tail_u64
                           : static_cast<uint64_t>(round_ops);
          meas_round_ops[r] = c * static_cast<uint64_t>(threads);
        }
      } else {
        for (size_t tid = 0; tid < threads; ++tid) {
          const auto &cnt_m = per_tid_round_counts_meas[tid];
          for (size_t r = 0; r < cnt_m.size(); ++r) {
            meas_round_ops[r] += static_cast<uint64_t>(cnt_m[r]);
          }
        }
      }
      meas_round_ops_prefix.resize(meas_round_ops.size() + 1, 0ULL);
      for (size_t i = 0; i < meas_round_ops.size(); ++i) {
        meas_round_ops_prefix[i + 1] = meas_round_ops_prefix[i] + meas_round_ops[i];
      }
    }

    std::cerr << "[concurrent_eval] index=" << index_type
              << " phase=warmup begin"
              << " warm_ops=" << warm_u64
              << " total_ops=" << nops_u64
              << std::endl;

#pragma omp parallel num_threads(static_cast<int>(threads))
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      Param param(threads, static_cast<uint32_t>(tid));
      PAYLOAD_TYPE tmp{};
      uint64_t local_scan_tuples = 0;
      std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> scan_buf;
      if (stream_mode == "scan") scan_buf.resize(scan_num);

      const uint64_t warm_full_rounds_local =
          (round_ops == 0) ? 0 : (warm_u64 / static_cast<uint64_t>(round_ops));
      const uint64_t warm_tail_local =
          (round_ops == 0) ? 0 : (warm_u64 % static_cast<uint64_t>(round_ops));

      if (is_pswix) {
        for (uint64_t r = 0; r < warm_full_rounds_local; ++r) {
          const uint64_t begin = r * static_cast<uint64_t>(round_ops);
          const uint64_t end = begin + static_cast<uint64_t>(round_ops);
          for (uint64_t pos = begin; pos < end; ++pos) {
            const ReplayOp &rop = pre.ops[static_cast<size_t>(pos)];
            exec_one(param, tmp, scan_buf, rop);
          }
          if (!heartbeat_wait) {
#pragma omp barrier
          }
        }
        {
          const uint64_t begin = warm_full_rounds_local * static_cast<uint64_t>(round_ops);
          const uint64_t end = begin + warm_tail_local;
          for (uint64_t pos = begin; pos < end; ++pos) {
            const ReplayOp &rop = pre.ops[static_cast<size_t>(pos)];
            exec_one(param, tmp, scan_buf, rop);
          }
          if (!heartbeat_wait) {
#pragma omp barrier
          }
        }
      } else {
        const std::vector<ReplayOp> &mine_ops = per_tid_ops[tid];
        const std::vector<uint32_t> &cnt_w = per_tid_round_counts_warm[tid];
        const size_t warm_end = per_tid_warm_end[tid];

        size_t op_i = 0;
        for (size_t r = 0; r < cnt_w.size(); ++r) {
          const uint32_t c = cnt_w[r];
          for (uint32_t j = 0; j < c; ++j) {
            exec_one(param, tmp, scan_buf, mine_ops[op_i]);
            ++op_i;
          }
          if (!heartbeat_wait) {
#pragma omp barrier
          }
        }
        INVARIANT(op_i == warm_end);
      }

      if (!heartbeat_wait) {
#pragma omp barrier
#pragma omp master
        {
          std::cerr << "[concurrent_eval] index=" << index_type
                    << " phase=warmup end"
                    << std::endl;
          t_start = std::chrono::steady_clock::now();
          std::cerr << "[concurrent_eval] index=" << index_type
                    << " phase=measured begin"
                    << std::endl;
        }
#pragma omp barrier
      } else {
        warm_done_cnt.fetch_add(1, std::memory_order_acq_rel);
        uint32_t spin = 0;
        while (warm_done_cnt.load(std::memory_order_acquire) < threads) {
          if (((++spin) & 0xFFu) == 0) {
            (void)index->get(heartbeat_key, tmp, &param);  // heartbeat to keep rcu_progress advancing
          } else {
            std::this_thread::yield();
          }
        }
        if (tid == 0) {
          std::cerr << "[concurrent_eval] index=" << index_type
                    << " phase=warmup end"
                    << std::endl;
          t_start = std::chrono::steady_clock::now();
          std::cerr << "[concurrent_eval] index=" << index_type
                    << " phase=measured begin"
                    << std::endl;
          measured_go.store(true, std::memory_order_release);
        } else {
          while (!measured_go.load(std::memory_order_acquire)) {
            (void)index->get(heartbeat_key, tmp, &param);  // heartbeat while waiting for measured start
            std::this_thread::yield();
          }
        }
      }

      const uint64_t rem_local = (nops_u64 > warm_u64) ? (nops_u64 - warm_u64) : 0ULL;
      const uint64_t meas_full_rounds_local =
          (round_ops == 0) ? 0 : (rem_local / static_cast<uint64_t>(round_ops));
      const uint64_t meas_tail_local =
          (round_ops == 0) ? 0 : (rem_local % static_cast<uint64_t>(round_ops));

      if (is_pswix) {
        auto maybe_time_exec = [&](uint64_t pos_u64, const ReplayOp &rop) {
          const bool sample = latency_sample ? (sample_mask[static_cast<size_t>(pos_u64)] != 0) : false;
          const bool do_time = sample;
          int64_t c0 = 0, c1 = 0;
          if (do_time) c0 = TSCNS::rdtsc();
          local_scan_tuples += exec_one(param, tmp, scan_buf, rop);
          if (do_time) {
            c1 = TSCNS::rdtsc();
            stats[tid].sample_cycles += static_cast<uint64_t>(c1 - c0);
            stats[tid].sample_ops += 1;
          }
        };

        for (uint64_t r = 0; r < meas_full_rounds_local; ++r) {
          const uint64_t begin = warm_u64 + r * static_cast<uint64_t>(round_ops);
          const uint64_t end = begin + static_cast<uint64_t>(round_ops);
          for (uint64_t pos = begin; pos < end; ++pos) {
            const ReplayOp &rop = pre.ops[static_cast<size_t>(pos)];
            maybe_time_exec(pos, rop);
          }
          if (!heartbeat_wait) {
#pragma omp barrier
          }
          if (seg_enabled && !seg_round_ends.empty() && !heartbeat_wait) {
#pragma omp master
            {
              const uint64_t round_end_excl = r + 1; // measured rounds are 0-based
              while (segment_recs.size() < seg_round_ends.size() &&
                     seg_round_ends[segment_recs.size()] == round_end_excl) {
                const uint64_t seg_start = (segment_recs.empty() ? 0ULL : segment_recs.back().round_end);
                const uint64_t seg_end = round_end_excl;
                SegmentRec rec;
                rec.seg_idx = segment_recs.size();
                rec.round_start = seg_start;
                rec.round_end = seg_end;
                const size_t b = static_cast<size_t>(seg_start);
                const size_t e = static_cast<size_t>(seg_end);
                rec.ops = meas_round_ops_prefix[e] - meas_round_ops_prefix[b];
                const auto now = std::chrono::steady_clock::now();
                rec.wall_end_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - t_start).count();
                segment_recs.push_back(std::move(rec));
              }
            }
#pragma omp barrier
          }
        }
        {
          const uint64_t begin = warm_u64 + meas_full_rounds_local * static_cast<uint64_t>(round_ops);
          const uint64_t end = begin + meas_tail_local;
          for (uint64_t pos = begin; pos < end; ++pos) {
            const ReplayOp &rop = pre.ops[static_cast<size_t>(pos)];
            maybe_time_exec(pos, rop);
          }
          if (!heartbeat_wait) {
#pragma omp barrier
          }
          if (seg_enabled && !seg_round_ends.empty() && !heartbeat_wait) {
#pragma omp master
            {
              const uint64_t round_end_excl = meas_full_rounds_local + 1ULL; // tail treated as final "round"
              while (segment_recs.size() < seg_round_ends.size() &&
                     seg_round_ends[segment_recs.size()] == round_end_excl) {
                const uint64_t seg_start = (segment_recs.empty() ? 0ULL : segment_recs.back().round_end);
                const uint64_t seg_end = round_end_excl;
                SegmentRec rec;
                rec.seg_idx = segment_recs.size();
                rec.round_start = seg_start;
                rec.round_end = seg_end;
                const size_t b = static_cast<size_t>(seg_start);
                const size_t e = static_cast<size_t>(seg_end);
                rec.ops = meas_round_ops_prefix[e] - meas_round_ops_prefix[b];
                const auto now = std::chrono::steady_clock::now();
                rec.wall_end_s = std::chrono::duration_cast<std::chrono::duration<double>>(now - t_start).count();
                segment_recs.push_back(std::move(rec));
              }
            }
#pragma omp barrier
          }
        }
      } else {
        const std::vector<ReplayOp> &mine_ops = per_tid_ops[tid];
        const std::vector<uint32_t> &cnt_w = per_tid_round_counts_warm[tid];
        const std::vector<uint32_t> &cnt_m = per_tid_round_counts_meas[tid];
        const size_t warm_end = per_tid_warm_end[tid];
        const std::vector<uint64_t> &bits = per_tid_sample_bits[tid];

        auto is_sampled = [&](size_t op_idx) -> bool {
          if (!latency_sample) return false;
          const size_t w = op_idx >> 6;
          if (w >= bits.size()) return false;
          return ((bits[w] >> (op_idx & 63ULL)) & 1ULL) != 0ULL;
        };

        size_t op_i = 0;
        for (size_t r = 0; r < cnt_w.size(); ++r) op_i += cnt_w[r];
        INVARIANT(op_i == warm_end);

#pragma omp barrier

        for (size_t r = 0; r < cnt_m.size(); ++r) {
          const uint32_t c = cnt_m[r];
          for (uint32_t j = 0; j < c; ++j) {
            const bool do_time = is_sampled(op_i);
            int64_t c0 = 0, c1 = 0;
            if (do_time) c0 = TSCNS::rdtsc();
            local_scan_tuples += exec_one(param, tmp, scan_buf, mine_ops[op_i]);
            if (do_time) {
              c1 = TSCNS::rdtsc();
              stats[tid].sample_cycles += static_cast<uint64_t>(c1 - c0);
              stats[tid].sample_ops += 1;
            }
            ++op_i;
          }
          const uint64_t round_end_excl = static_cast<uint64_t>(r + 1);
          if (seg_enabled && !seg_round_ends.empty() && heartbeat_wait) {
            INVARIANT(hb_round_barrier != nullptr);
            auto heartbeat_fn = [&]() { (void)index->get(heartbeat_key, tmp, &param); };
            hb_round_barrier->wait(
                heartbeat_fn,
                [&]() {
                  while (segment_recs.size() < seg_round_ends.size() &&
                         seg_round_ends[segment_recs.size()] == round_end_excl) {
                    const uint64_t seg_start = (segment_recs.empty() ? 0ULL : segment_recs.back().round_end);
                    const uint64_t seg_end = round_end_excl;
                    SegmentRec rec;
                    rec.seg_idx = segment_recs.size();
                    rec.round_start = seg_start;
                    rec.round_end = seg_end;
                    const size_t b = static_cast<size_t>(seg_start);
                    const size_t e = static_cast<size_t>(seg_end);
                    rec.ops = meas_round_ops_prefix[e] - meas_round_ops_prefix[b];
                    const auto now = std::chrono::steady_clock::now();
                    rec.wall_end_s =
                        std::chrono::duration_cast<std::chrono::duration<double>>(now - t_start).count();
                    segment_recs.push_back(std::move(rec));
                  }
                });
          } else {
            if (!heartbeat_wait) {
#pragma omp barrier
            }
            if (seg_enabled && !seg_round_ends.empty() && !heartbeat_wait) {
#pragma omp master
              {
                while (segment_recs.size() < seg_round_ends.size() &&
                       seg_round_ends[segment_recs.size()] == round_end_excl) {
                  const uint64_t seg_start = (segment_recs.empty() ? 0ULL : segment_recs.back().round_end);
                  const uint64_t seg_end = round_end_excl;
                  SegmentRec rec;
                  rec.seg_idx = segment_recs.size();
                  rec.round_start = seg_start;
                  rec.round_end = seg_end;
                  const size_t b = static_cast<size_t>(seg_start);
                  const size_t e = static_cast<size_t>(seg_end);
                  rec.ops = meas_round_ops_prefix[e] - meas_round_ops_prefix[b];
                  const auto now = std::chrono::steady_clock::now();
                  rec.wall_end_s =
                      std::chrono::duration_cast<std::chrono::duration<double>>(now - t_start).count();
                  segment_recs.push_back(std::move(rec));
                }
              }
#pragma omp barrier
            }
          }
        }
      }

      if (!heartbeat_wait) {
#pragma omp barrier
#pragma omp master
        {
          t_end = std::chrono::steady_clock::now();
          wall_meas_s =
              std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
          std::cerr << "[concurrent_eval] index=" << index_type
                    << " phase=measured end"
                    << " wall_measured_s=" << std::fixed << std::setprecision(6) << wall_meas_s
                    << std::defaultfloat
                    << std::endl;
        }
#pragma omp barrier
      } else {
        meas_done_cnt.fetch_add(1, std::memory_order_acq_rel);
        uint32_t spin = 0;
        while (meas_done_cnt.load(std::memory_order_acquire) < threads) {
          if (((++spin) & 0xFFu) == 0) {
            (void)index->get(heartbeat_key, tmp, &param);  // heartbeat to avoid RCU starvation while waiting
          } else {
            std::this_thread::yield();
          }
        }
        if (!measured_end_recorded.exchange(true, std::memory_order_acq_rel)) {
          t_end = std::chrono::steady_clock::now();
          wall_meas_s =
              std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
          std::cerr << "[concurrent_eval] index=" << index_type
                    << " phase=measured end"
                    << " wall_measured_s=" << std::fixed << std::setprecision(6) << wall_meas_s
                    << std::defaultfloat
                    << std::endl;
        }
      }
      stats[tid].scan_tuples = local_scan_tuples;
    } // end omp parallel

    uint64_t sample_cycles_sum = 0;
    uint64_t sample_ops_sum = 0;
    uint64_t scan_tuples_sum = 0;
    for (const auto &st : stats) {
      sample_cycles_sum += st.sample_cycles;
      sample_ops_sum += st.sample_ops;
      scan_tuples_sum += st.scan_tuples;
    }

    uint64_t est_total_thread_cycles = 0;
    double est_total_thread_s = 0.0;
    if (latency_sample) {
      const double p = latency_sample_ratio;
      est_total_thread_cycles =
          static_cast<uint64_t>(std::llround(static_cast<double>(sample_cycles_sum) / p));
      TSCNS tn;
      tn.init();
      est_total_thread_s = (double)tn.tsc2ns_delta(static_cast<int64_t>(est_total_thread_cycles)) / 1e9;
    }

    const long long memory_bytes = seg_enabled ? 0LL : index->memory_consumption();
    const std::string data_name = std::filesystem::path(keys_file_path).filename().string();
    std::ostringstream line;
    line << "Algorithm=" << index_type
         << ";Data=" << data_name
         << ";TimeWindow=" << time_window
         << ";ScanNum=" << scan_num
         << ";Memory=" << memory_bytes
         << ";WallTimeMeasured=" << std::fixed << std::setprecision(6) << wall_meas_s << ";"
         << std::defaultfloat;
    line << config_kv_string();
    line << "SampleOps=" << sample_ops_sum << ";";
    line << "SampleCyclesSum=" << sample_cycles_sum << ";";
    if (latency_sample) {
      line << "ThreadCyclesEst=" << est_total_thread_cycles << ";";
      line << "ThreadTimeEst=" << std::fixed << std::setprecision(6) << est_total_thread_s << ";"
           << std::defaultfloat;
    }
    if (stream_mode == "scan") {
      line << "ScanTuples=" << scan_tuples_sum << ";";
    }

    if (!seg_enabled) {
      std::ofstream ofile(out_path, std::ios::app);
      ofile << line.str() << std::endl;
      ofile.close();
    } else {
      const uint64_t max_round_end = max_round_end_u64;
      // Always append a final trailing segment if needed.
      if (segment_recs.empty() || segment_recs.back().round_end < max_round_end) {
        SegmentRec rec;
        rec.seg_idx = segment_recs.size();
        rec.round_start = segment_recs.empty() ? 0ULL : segment_recs.back().round_end;
        rec.round_end = max_round_end;
        const size_t b = static_cast<size_t>(rec.round_start);
        const size_t e = static_cast<size_t>(rec.round_end);
        rec.ops = meas_round_ops_prefix[e] - meas_round_ops_prefix[b];
        rec.wall_end_s = wall_meas_s; // end of measured window
        segment_recs.push_back(std::move(rec));
      }

      std::ofstream ofile(out_path, std::ios::app);
      double prev_end_s = 0.0;
      for (const auto &rec : segment_recs) {
        const double seg_wall_s = rec.wall_end_s - prev_end_s;
        prev_end_s = rec.wall_end_s;
        std::ostringstream seg;
        seg << "Algorithm=" << index_type
            << ";Data=" << data_name
            << ";TimeWindow=" << time_window
            << ";ScanNum=" << scan_num
            << ";SegmentIdx=" << rec.seg_idx
            << ";SegmentRoundStart=" << rec.round_start
            << ";SegmentRoundEnd=" << rec.round_end
            << ";SegmentOps=" << rec.ops
            << ";SegmentWallTimeMeasured=" << std::fixed << std::setprecision(6) << seg_wall_s << ";"
            << std::defaultfloat;
        seg << config_kv_string();
        if (seg_wall_s > 0.0) {
          seg << "Throughput=" << (static_cast<double>(rec.ops) / seg_wall_s) << ";";
        }
        ofile << seg.str() << std::endl;
      }
      ofile.close();
    }

    delete index;
  }
};

}