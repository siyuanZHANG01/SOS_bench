#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "../tscns.h"
#include "./flags.h"
#include "./utils.h"
#include "../competitor/competitor.h"

template<typename KEY_TYPE, typename PAYLOAD_TYPE>
class StreamingBenchmark {
  using index_t = indexInterface<KEY_TYPE, PAYLOAD_TYPE>;

  struct Precomputed;
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

  std::string index_type;
  std::vector<std::string> all_index_type;
  std::string keys_file_path;
  std::string keys_file_type;
  std::string ts_file_path;
  std::string ts_file_type;

  long long table_size = -1;
  size_t time_window = 1000000;
  std::string stream_mode = "lookup";
  size_t scan_num = 100;
  size_t random_seed = 1;
  double rw_ratio = 1.0;
  double query_hit_ratio = 1.0;
  std::string query_distribution = "uniform"; // uniform | zipf
  std::string zipf_generator = "unscrambled"; // unscrambled | scrambled
  std::string output_path = "./out_streaming.txt";
  bool pre_run = false;
  std::string pre_run_path = ""; 
  std::string segment_file_path = ""; 
  std::string segment_by = "ts";
  bool latency_sample = false;
  double latency_sample_ratio = 0.01;
  bool latency_time_all_ops = false;
  bool warmup_check = false; 
  size_t warm_up = 0; 
  size_t progress_every_ops = 0;
  bool shuffle_keys = false;
  bool half_range = false;
  bool stronger_non_duplicate = false;
  size_t stronger_non_duplicate_dropped_ = 0;
  size_t stronger_non_duplicate_remaining_ = 0;

  std::vector<DatasetPair> dataset_pairs_;
  std::vector<std::string> all_stream_mode_;
  std::vector<size_t> all_scan_num_;
  std::vector<size_t> all_random_seed_;
  std::vector<double> all_rw_ratio_;
  std::vector<double> all_query_hit_ratio_;
  std::vector<std::string> all_query_distribution_;
  std::vector<std::string> all_pre_run_path_;
  std::vector<std::string> all_segment_file_path_;

  KEY_TYPE *keys_ = nullptr;
  PAYLOAD_TYPE *ts_ = nullptr;
  long long n_ = 0;
  uint64_t first_ts_ = 0; 
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> init_kv_sorted_; 

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

  std::string result_file_path_for_config() const {
    std::ostringstream sig;
    sig << keys_file_path << "|"
        << ts_file_path << "|"
        << keys_file_type << "|"
        << ts_file_type << "|"
        << "tw=" << time_window << "|"
        << "sm=" << stream_mode << "|"
        << "scan=" << scan_num << "|"
        << "rw=" << std::setprecision(17) << rw_ratio << "|"
        << "hr=" << std::setprecision(17) << query_hit_ratio << "|"
        << "qd=" << query_distribution << "|"
        << "zg=" << zipf_generator << "|"
        << "shuf=" << (shuffle_keys ? 1 : 0) << "|"
        << "half=" << (half_range ? 1 : 0) << "|"
        << "snd=" << (stronger_non_duplicate ? 1 : 0) << "|"
        << "seg=" << segment_file_path << "|"
        << "segby=" << segment_by << "|"
        << "pre=" << pre_run_path << "|"
        << "ls=" << (latency_sample ? 1 : 0) << "|"
        << "lsr=" << std::setprecision(17) << latency_sample_ratio;
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
    out << "SegmentFile=" << segment_file_path << ";";
    out << "SegmentBy=" << segment_by << ";";
    out << "LatencySample=" << (latency_sample ? 1 : 0) << ";";
    out << "LatencySampleRatio=" << std::setprecision(17) << latency_sample_ratio << ";";
    out << "LatencyTimeAllOps=" << (latency_time_all_ops ? 1 : 0) << ";";
    out << "WarmUp=" << warm_up << ";";
    out << "ShuffleKeys=" << (shuffle_keys ? 1 : 0) << ";";
    out << "HalfRange=" << (half_range ? 1 : 0) << ";";
    out << "StrongerNonDuplicate=" << (stronger_non_duplicate ? 1 : 0) << ";";
    out << std::defaultfloat;
    return out.str();
  }

  void parse_args(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);
    auto it_k = flags.find("keys_file");
    if (it_k == flags.end()) {
      std::cout << "Required flag --keys_file was not found. Raw argv:" << std::endl;
      for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "]=" << argv[i] << std::endl;
      }
      std::cout << "Parsed flag keys:" << std::endl;
      for (const auto &kv : flags) {
        std::cout << "  '" << kv.first << "'" << std::endl;
      }
      COUT_N_EXIT("Required flag --keys_file was not found");
    }
    std::cout << "keys_file = " << it_k->second << std::endl;
    const std::vector<std::string> keys_files = split_csv(it_k->second);
    INVARIANT(!keys_files.empty());
    std::vector<std::string> ts_files;
    auto it_t = flags.find("ts_file");
    if (it_t == flags.end() || it_t->second.empty()) {
      std::cout << "ts_file = (generated)" << std::endl;
      ts_files.clear();
    } else {
      std::cout << "ts_file = " << it_t->second << std::endl;
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
    keys_file_type = get_with_default(flags, "keys_file_type", "binary");
    ts_file_path = dataset_pairs_[0].ts_file;
    ts_file_type = get_with_default(flags, "ts_file_type", keys_file_type);
    table_size = std::stoll(get_with_default(flags, "table_size", "-1"));
    time_window = static_cast<size_t>(std::stoull(get_with_default(flags, "time_window", "1000000")));
    {
      const auto v = split_csv(get_with_default(flags, "stream_mode", "lookup"));
      all_stream_mode_ = v;
      stream_mode = all_stream_mode_[0];
    }
    {
      const auto v = split_csv(get_with_default(flags, "scan_num", "100"));
      all_scan_num_.clear();
      all_scan_num_.reserve(v.size());
      for (const auto &s : v) all_scan_num_.push_back(static_cast<size_t>(std::stoull(s)));
      scan_num = all_scan_num_[0];
    }
    {
      const auto v = split_csv(get_with_default(flags, "seed", "1"));
      all_random_seed_.clear();
      all_random_seed_.reserve(v.size());
      for (const auto &s : v) all_random_seed_.push_back(static_cast<size_t>(std::stoull(s)));
      random_seed = all_random_seed_[0];
    }
    {
      const auto v = split_csv(get_with_default(flags, "rw_ratio", "1.0"));
      all_rw_ratio_.clear();
      all_rw_ratio_.reserve(v.size());
      for (const auto &s : v) all_rw_ratio_.push_back(std::stod(s));
      rw_ratio = all_rw_ratio_[0];
    }
    {
      const auto v = split_csv(get_with_default(flags, "query_hit_ratio", "1.0"));
      all_query_hit_ratio_.clear();
      all_query_hit_ratio_.reserve(v.size());
      for (const auto &s : v) all_query_hit_ratio_.push_back(std::stod(s));
      query_hit_ratio = all_query_hit_ratio_[0];
    }
    {
      const auto v = split_csv(get_with_default(flags, "query_distribution", "uniform"));
      all_query_distribution_ = v;
      query_distribution = all_query_distribution_[0];
    }
    zipf_generator = get_with_default(flags, "zipf_generator", "unscrambled");
    output_path = get_with_default(flags, "output_path", "./out_streaming.txt");
    pre_run = get_boolean_flag(flags, "pre_run");
    all_index_type = get_comma_separated(flags, "index");
    if (all_index_type.empty()) {
      if (pre_run) {
        all_index_type = {"btree"};
      } else {
        COUT_N_EXIT("Required flag --index was not found (e.g., --index=btree,swix)");
      }
    }
    index_type = all_index_type[0];
    {
      const auto v = split_csv(get_with_default(flags, "pre_run_path", ""));
      all_pre_run_path_ = v;
      pre_run_path = all_pre_run_path_[0];
    }
    {
      const auto v = split_csv(get_with_default(flags, "segment_file", ""));
      all_segment_file_path_ = v;
      segment_file_path = all_segment_file_path_[0];
    }
    segment_by = get_with_default(flags, "segment_by", "ts");
    latency_sample = get_boolean_flag(flags, "latency_sample");
    latency_sample_ratio = std::stod(get_with_default(flags, "latency_sample_ratio", "0.01"));
    latency_time_all_ops = get_boolean_flag(flags, "latency_time_all_ops");
    warmup_check = get_boolean_flag(flags, "warmup_check");
    warm_up = static_cast<size_t>(std::stoull(get_with_default(flags, "warm_up", "0")));
    progress_every_ops = static_cast<size_t>(std::stoull(get_with_default(flags, "progress_every_ops", "0")));
    shuffle_keys = get_boolean_flag(flags, "shuffle_keys");
    half_range = get_boolean_flag(flags, "half_range");
    {
      const bool a = (flags.find("stronger_non_duplicate") != flags.end());
      const bool b = (flags.find("stronger_non-duplicate") != flags.end());
      stronger_non_duplicate = (a || b);
      std::cout << "stronger_non_duplicate = " << (stronger_non_duplicate ? 1 : 0) << std::endl;
    }

    INVARIANT(time_window > 0);
    for (const auto &m : all_stream_mode_) INVARIANT(m == "lookup" || m == "scan");
    for (const auto &r : all_rw_ratio_) INVARIANT(r >= 0.0);
    for (const auto &hr : all_query_hit_ratio_) {
      INVARIANT(hr >= 0.0);
      INVARIANT(hr <= 1.0);
    }
    for (const auto &qd : all_query_distribution_) INVARIANT(qd == "uniform" || qd == "zipf");
    INVARIANT(segment_by == "ts" || segment_by == "op");
    for (const auto &m : all_stream_mode_) {
      if (m == "scan") {
        for (const auto &sn : all_scan_num_) INVARIANT(sn > 0);
      }
    }
    if (query_distribution == "zipf" || std::find(all_query_distribution_.begin(), all_query_distribution_.end(), "zipf") != all_query_distribution_.end()) {
      INVARIANT(zipf_generator == "unscrambled" || zipf_generator == "scrambled");
    }
    if (latency_sample) {
      INVARIANT(latency_sample_ratio > 0.0);
      INVARIANT(latency_sample_ratio <= 1.0);
    }
    if (warmup_check) {
      latency_sample = true;
      INVARIANT(latency_sample_ratio > 0.0);
      INVARIANT(latency_sample_ratio <= 1.0);
    }
    if (shuffle_keys && !ts_file_path.empty()) {
      std::cerr << "[streaming] warning: --shuffle_keys is enabled with an explicit ts_file; "
                   "shuffling keys only (timestamps unchanged)."
                << std::endl;
    }
  }

  void run() {
    for (size_t di = 0; di < dataset_pairs_.size(); ++di) {
      keys_file_path = dataset_pairs_[di].keys_file;
      ts_file_path = dataset_pairs_[di].ts_file;

      if (pre_run) {
        load_inputs();
        const size_t init_end = initial_end_idx();
        first_ts_ = static_cast<uint64_t>(ts_[0]);
        init_kv_sorted_.clear();
        init_kv_sorted_.reserve(init_end);
        for (size_t i = 0; i < init_end; ++i) {
          init_kv_sorted_.emplace_back(keys_[i], ts_[i]);
        }
        std::sort(init_kv_sorted_.begin(), init_kv_sorted_.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
      }

      const bool pre_path_zipped = (all_pre_run_path_.size() == dataset_pairs_.size());
      const bool seg_path_zipped = (all_segment_file_path_.size() == dataset_pairs_.size());

      const std::vector<std::string> pre_paths =
          pre_path_zipped ? std::vector<std::string>{all_pre_run_path_[di]} : all_pre_run_path_;
      const std::vector<std::string> seg_paths =
          seg_path_zipped ? std::vector<std::string>{all_segment_file_path_[di]} : all_segment_file_path_;

      for (const auto &sm : all_stream_mode_) {
        stream_mode = sm;
        for (const auto &sn : all_scan_num_) {
          scan_num = sn;
          for (const auto &qd : all_query_distribution_) {
            query_distribution = qd;
            for (const auto &rw : all_rw_ratio_) {
              rw_ratio = rw;
              for (const auto &hr : all_query_hit_ratio_) {
                query_hit_ratio = hr;
                for (const auto &seg : seg_paths) {
                  segment_file_path = seg;
                  for (const auto &prp : pre_paths) {
                    pre_run_path = prp;
                    const std::string cfg_out_path = result_file_path_for_config();
                    for (const auto &seed : all_random_seed_) {
                      random_seed = seed;
                      if (pre_run) {
                        Precomputed pre = precompute_sequences();
                        const std::string bin_path = pre_run_file_path();
                        write_pre_run_file(pre, bin_path);
                        write_pre_run_config_file(pre, pre_run_config_path());
                        continue;
                      }
                      if (warmup_check) {
                        Precomputed pre = read_pre_run_file(pre_run_file_path());
                        const size_t init_end = static_cast<size_t>(pre.init_end);
                        INVARIANT(pre.time_window == static_cast<uint64_t>(time_window));
                        first_ts_ = !pre.insert_ops.empty() ? pre.insert_ops[0].ts : 0ULL;
                        init_kv_sorted_.clear();
                        init_kv_sorted_.reserve(init_end);
                        for (size_t i = 0; i < init_end; ++i) {
                          init_kv_sorted_.emplace_back(pre.insert_ops[i].key, static_cast<PAYLOAD_TYPE>(pre.insert_ops[i].ts));
                        }
                        std::sort(init_kv_sorted_.begin(), init_kv_sorted_.end(),
                                  [](const auto &a, const auto &b) { return a.first < b.first; });
                        build_replay_ops(pre);
                        std::vector<InsertOp>().swap(pre.insert_ops);
                        std::vector<QueryOp>().swap(pre.query_ops);
                        for (const auto &idx_type : all_index_type) {
                          index_type = idx_type;
                          run_one_index(pre, init_end, cfg_out_path, /*warmup_check_mode=*/true);
                        }
                        continue;
                      }

                      Precomputed pre = read_pre_run_file(pre_run_file_path());
                      INVARIANT(pre.time_window == static_cast<uint64_t>(time_window));
                      const size_t init_end = static_cast<size_t>(pre.init_end);

                      first_ts_ = !pre.insert_ops.empty() ? pre.insert_ops[0].ts : 0ULL;
                      init_kv_sorted_.clear();
                      init_kv_sorted_.reserve(init_end);
                      for (size_t i = 0; i < init_end; ++i) {
                        init_kv_sorted_.emplace_back(pre.insert_ops[i].key, static_cast<PAYLOAD_TYPE>(pre.insert_ops[i].ts));
                      }
                      std::sort(init_kv_sorted_.begin(), init_kv_sorted_.end(),
                                [](const auto &a, const auto &b) { return a.first < b.first; });
                      build_replay_ops(pre);
                      std::vector<InsertOp>().swap(pre.insert_ops);
                      std::vector<QueryOp>().swap(pre.query_ops);

                      for (const auto &idx_type : all_index_type) {
                        index_type = idx_type;
                        run_one_index(pre, init_end, cfg_out_path, /*warmup_check_mode=*/false);
                      }
                      init_kv_sorted_.clear();
                    }
                  }
                }
              }
            }
          }
        }
      }

      if (pre_run) {
        cleanup_inputs();
      }
      init_kv_sorted_.clear();
      init_kv_sorted_.shrink_to_fit();
      first_ts_ = 0;
    }
  }

private:
  void run_one_index(const Precomputed &pre, size_t init_end, const std::string &out_path, bool warmup_check_mode) {
    index_t *index = prepare_index();

    TSCNS tn;
    tn.init();

    bool meas_started = (warm_up == 0);
    int64_t run_c0 = 0;
    int64_t seg_c0 = 0;
    int64_t meas_c0 = 0;
    int64_t seg_meas_c0 = 0;

    std::vector<uint64_t> total_lat_cy;
    std::vector<uint64_t> query_lat_cy;
    std::vector<uint64_t> insert_lat_cy;
    std::vector<uint64_t> delete_lat_cy;

    if (latency_sample) {
      const uint64_t del_cnt = pre.del_ops_n;
      const double approx_total =
          static_cast<double>(pre.query_ops_n) +
          static_cast<double>(pre.ins_ops_n) +
          static_cast<double>(del_cnt);
      const size_t approx_samples =
          static_cast<size_t>(std::max<double>(1.0, approx_total * latency_sample_ratio));
      const size_t reserve_samples = approx_samples + std::max<size_t>(8, approx_samples / 8);
      query_lat_cy.reserve(reserve_samples);
      insert_lat_cy.reserve(reserve_samples);
      delete_lat_cy.reserve(reserve_samples);
      total_lat_cy.reserve(reserve_samples * 3);
    }

    Param param(1, 0);
    std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> scan_result;
    if (stream_mode == "scan") scan_result.resize(scan_num);

    uint64_t totalCount = 0;
    uint64_t scanTuples = 0;
    uint64_t searchCycles = 0;
    uint64_t insertCycles = 0;
    uint64_t deleteCycles = 0;
    uint64_t searchOps = 0;
    uint64_t insertOps = 0;
    uint64_t deleteOps = 0;
    struct SegmentRec {
      size_t seg_idx = 0;
      uint64_t seg_start = 0;
      uint64_t seg_end = 0;
      long long mem_bytes = 0;
      int64_t seg_c0 = 0;
      int64_t seg_c1 = 0;
      bool meas_valid = false;
      int64_t seg_meas_c0 = 0;
      int64_t seg_meas_c1 = 0;

      bool is_scan_mode = false;
      uint64_t scanTuples = 0;
      uint64_t totalCount = 0;
      uint64_t searchCycles = 0;
      uint64_t insertCycles = 0;
      uint64_t deleteCycles = 0;
      uint64_t searchOps = 0;
      uint64_t insertOps = 0;
      uint64_t deleteOps = 0;

      size_t total_lat_b = 0, total_lat_e = 0;
      size_t query_lat_b = 0, query_lat_e = 0;
      size_t insert_lat_b = 0, insert_lat_e = 0;
      size_t delete_lat_b = 0, delete_lat_e = 0;
    };
    std::vector<SegmentRec> segment_recs;

    const std::vector<uint64_t> cuts = read_segment_cuts(segment_file_path);
    if (!cuts.empty()) {
      segment_recs.reserve(cuts.size() + 1);
    }
    if (!warmup_check_mode && warm_up > 0) {
      const uint64_t need = static_cast<uint64_t>(warm_up) * 2ULL;
      auto ops_upto = [&](uint64_t x) -> uint64_t {
        ReplayOp k;
        k.ts = x;
        k.type = 2;
        const auto it = std::upper_bound(
            pre.ops.begin(), pre.ops.end(), k,
            [](const ReplayOp &a, const ReplayOp &b) {
              if (a.ts != b.ts) return a.ts < b.ts;
              return a.type < b.type;
            });
        return static_cast<uint64_t>(it - pre.ops.begin());
      };
      if (cuts.empty()) {
        const uint64_t tot = static_cast<uint64_t>(pre.ops.size());
        if (tot < need) {
          COUT_N_EXIT("[streaming] warm_up too large: total ops < warm_up*2");
        }
      } else {
        uint64_t seg0 = 0;
        if (segment_by == "ts") {
          const uint64_t first_cut = cuts.front();
          seg0 = ops_upto(first_cut);
        } else {
          seg0 = cuts.front();
        }
        if (seg0 < need) {
          COUT_N_EXIT("[streaming] warm_up too large: first segment ops < warm_up*2");
        }
      }
    }
    size_t cut_idx = 0;
    const uint64_t end_ts = pre.end_ts;
    uint64_t seg_start = 0ULL;
    if (segment_by == "ts") {
      seg_start = !pre.ops.empty() ? pre.ops.front().ts : 0ULL;
    }

    if (segment_by == "op") {
      for (const uint64_t c : cuts) {
        INVARIANT(c > 0);
      }
    }

    std::vector<std::pair<uint64_t, uint64_t>> total_lat_pts;
    if (warmup_check_mode) total_lat_pts.reserve(total_lat_cy.capacity());

    std::vector<uint8_t> sample_mask;
    if (latency_sample) {
      sample_mask.resize(pre.ops.size(), 0);
      std::ostringstream ss;
      ss << "latency_sample_mask|"
         << "sig=" << hex_u64(pre_run_sig_hash()) << "|"
         << "seed=" << random_seed << "|"
         << "ratio=" << std::setprecision(17) << latency_sample_ratio;
      std::mt19937_64 rng(fnv1a64(ss.str()));
      std::bernoulli_distribution bern(latency_sample_ratio);
      for (size_t i = 0; i < pre.ops.size(); ++i) {
        sample_mask[i] = static_cast<uint8_t>(bern(rng) ? 1 : 0);
      }
    }

    uint64_t q_done = 0, ins_done = 0, del_done = 0;

    run_c0 = tn.rdtsc();
    seg_c0 = run_c0;
    meas_c0 = run_c0;
    seg_meas_c0 = run_c0;

    size_t seg_total_b = 0, seg_query_b = 0, seg_insert_b = 0, seg_delete_b = 0;

    uint64_t op_i = 0;
    for (size_t op_pos = 0; op_pos < pre.ops.size(); ++op_pos) {
      const ReplayOp &rop = pre.ops[op_pos];
      const uint64_t t = rop.ts;
      const uint64_t cur_i = op_i++;
      if (progress_every_ops > 0 && (cur_i % static_cast<uint64_t>(progress_every_ops) == 0ULL)) {
        std::cout << "[streaming][progress] index=" << index_type
                  << " op_i=" << cur_i
                  << " t=" << t
                  << " q_done=" << q_done << "/" << pre.query_ops_n
                  << " ins_done=" << ins_done << "/" << pre.ins_ops_n
                  << " del_done=" << del_done << "/" << pre.del_ops_n
                  << std::endl;
      }
      if (!warmup_check_mode && !meas_started && warm_up > 0 && cur_i >= static_cast<uint64_t>(warm_up)) {
        meas_c0 = tn.rdtsc();
        seg_meas_c0 = meas_c0;
        meas_started = true;
      }
      const bool in_measure = warmup_check_mode || meas_started;

      auto should_flush_segment = [&]() -> bool {
        if (cut_idx >= cuts.size()) return false;
        if (segment_by == "ts") {
          return t > cuts[cut_idx];
        }
        return cur_i >= cuts[cut_idx];
      };

      while (!warmup_check_mode && should_flush_segment()) {
        const int64_t now_c_pre = tn.rdtsc();
        SegmentRec rec;
        rec.seg_idx = cut_idx;
        rec.seg_start = seg_start;
        rec.seg_end = cuts[cut_idx];
        rec.mem_bytes = index->memory_consumption();
        const int64_t now_c_post = tn.rdtsc();
        rec.seg_c0 = seg_c0;
        rec.seg_c1 = now_c_pre;
        rec.meas_valid = meas_started;
        rec.seg_meas_c0 = seg_meas_c0;
        rec.seg_meas_c1 = now_c_pre;

        rec.is_scan_mode = (stream_mode == "scan");
        rec.scanTuples = scanTuples;
        rec.totalCount = totalCount;
        rec.searchCycles = searchCycles;
        rec.insertCycles = insertCycles;
        rec.deleteCycles = deleteCycles;
        rec.searchOps = searchOps;
        rec.insertOps = insertOps;
        rec.deleteOps = deleteOps;
        rec.total_lat_b = seg_total_b;  rec.total_lat_e = total_lat_cy.size();  seg_total_b = rec.total_lat_e;
        rec.query_lat_b = seg_query_b;  rec.query_lat_e = query_lat_cy.size();  seg_query_b = rec.query_lat_e;
        rec.insert_lat_b = seg_insert_b; rec.insert_lat_e = insert_lat_cy.size(); seg_insert_b = rec.insert_lat_e;
        rec.delete_lat_b = seg_delete_b; rec.delete_lat_e = delete_lat_cy.size(); seg_delete_b = rec.delete_lat_e;
        segment_recs.push_back(std::move(rec));

        totalCount = 0;
        scanTuples = 0;
        searchCycles = insertCycles = deleteCycles = 0;
        searchOps = insertOps = deleteOps = 0;
        if (meas_started) {
          seg_meas_c0 = now_c_post;
        }
        seg_c0 = now_c_post;
        seg_start = cuts[cut_idx];
        ++cut_idx;
      }

      param.ts = t;

      if (rop.type == 0) {
        const bool sample = latency_sample ? (sample_mask[op_pos] != 0) : false;
        int64_t b0 = 0, b1 = 0;
        const bool do_time = in_measure && (!latency_sample || latency_time_all_ops || sample);
        if (do_time) b0 = tn.rdtsc();
        if (!(index_type == "swix" || index_type == "imtree")) {
          (void)index->remove(static_cast<KEY_TYPE>(rop.key), &param);
        }
        if (do_time) b1 = tn.rdtsc();
        if (in_measure) ++deleteOps;
        if (!latency_sample) {
          if (in_measure) deleteCycles += (b1 - b0);
        } else if (sample && in_measure) {
          const uint64_t d_cy = static_cast<uint64_t>(b1 - b0);
          delete_lat_cy.push_back(d_cy);
          total_lat_cy.push_back(d_cy);
          if (warmup_check_mode) total_lat_pts.emplace_back(cur_i, d_cy);
        }
        ++del_done;
      } else if (rop.type == 1) {
        const bool sample = latency_sample ? (sample_mask[op_pos] != 0) : false;
        if (stream_mode == "lookup") {
          PAYLOAD_TYPE tmp{};
          int64_t s0 = 0, s1 = 0;
          const bool do_time = in_measure && (!latency_sample || latency_time_all_ops || sample);
          if (do_time) s0 = tn.rdtsc();
          bool found = index->get(static_cast<KEY_TYPE>(rop.key), tmp, &param);
          if (do_time) s1 = tn.rdtsc();
          if (in_measure) ++searchOps;
          if (!latency_sample) {
            if (in_measure) {
              searchCycles += (s1 - s0);
              totalCount += static_cast<uint64_t>(found);
            }
          } else if (sample && in_measure) {
            const uint64_t q_cy = static_cast<uint64_t>(s1 - s0);
            query_lat_cy.push_back(q_cy);
            total_lat_cy.push_back(q_cy);
            if (warmup_check_mode) total_lat_pts.emplace_back(cur_i, q_cy);
          }
        } else {
          param.scan_end_key = (index_type == "swix" || index_type == "imtree" || index_type == "dili")
                                   ? static_cast<uint64_t>(rop.aux)
                                   : 0ULL;
          int64_t s0 = 0, s1 = 0;
          const bool do_time = in_measure && (!latency_sample || latency_time_all_ops || sample);
          if (do_time) s0 = tn.rdtsc();
          auto got = index->scan(static_cast<KEY_TYPE>(rop.key), scan_num, scan_result.data(), &param);
          if (do_time) s1 = tn.rdtsc();
          if (in_measure) {
            scanTuples += static_cast<uint64_t>(got);
            ++searchOps;
          }
          if (!latency_sample) {
            if (in_measure) {
              searchCycles += (s1 - s0);
              totalCount += static_cast<uint64_t>(got);
            }
          } else if (sample && in_measure) {
            const uint64_t q_cy = static_cast<uint64_t>(s1 - s0);
            query_lat_cy.push_back(q_cy);
            total_lat_cy.push_back(q_cy);
            if (warmup_check_mode) total_lat_pts.emplace_back(cur_i, q_cy);
          }
        }
        ++q_done;
      } else {
        const bool sample = latency_sample ? (sample_mask[op_pos] != 0) : false;
        int64_t i0 = 0, i1 = 0;
        const bool do_time = in_measure && (!latency_sample || latency_time_all_ops || sample);
        if (do_time) i0 = tn.rdtsc();
        (void)index->put(static_cast<KEY_TYPE>(rop.key),
                         static_cast<PAYLOAD_TYPE>(rop.ts),
                         &param);
        if (do_time) i1 = tn.rdtsc();
        if (in_measure) ++insertOps;
        if (!latency_sample) {
          if (in_measure) insertCycles += (i1 - i0);
        } else if (sample && in_measure) {
          const uint64_t i_cy = static_cast<uint64_t>(i1 - i0);
          insert_lat_cy.push_back(i_cy);
          total_lat_cy.push_back(i_cy);
          if (warmup_check_mode) total_lat_pts.emplace_back(cur_i, i_cy);
        }
        ++ins_done;
      }
    }

    const int64_t run_c1 = tn.rdtsc();
    const double wall_s = (double)tn.tsc2ns_delta(run_c1 - run_c0) / 1e9;
    const double wall_meas_s =
        (!warmup_check_mode && warm_up > 0 && !meas_started) ? 0.0
        : (double)tn.tsc2ns_delta(run_c1 - meas_c0) / 1e9;

    const uint64_t t_last = pre.end_ts;
    if (!warmup_check_mode && !cuts.empty() && end_ts > 0) {
      SegmentRec rec;
      rec.seg_idx = cut_idx;
      rec.seg_start = seg_start;
      rec.seg_end = (segment_by == "ts") ? t_last : static_cast<uint64_t>(pre.ops.size());
      rec.mem_bytes = index->memory_consumption();
      rec.seg_c0 = seg_c0;
      rec.seg_c1 = run_c1;
      rec.meas_valid = meas_started;
      rec.seg_meas_c0 = seg_meas_c0;
      rec.seg_meas_c1 = run_c1;

      rec.is_scan_mode = (stream_mode == "scan");
      rec.scanTuples = scanTuples;
      rec.totalCount = totalCount;
      rec.searchCycles = searchCycles;
      rec.insertCycles = insertCycles;
      rec.deleteCycles = deleteCycles;
      rec.searchOps = searchOps;
      rec.insertOps = insertOps;
      rec.deleteOps = deleteOps;
      rec.total_lat_b = seg_total_b;  rec.total_lat_e = total_lat_cy.size();
      rec.query_lat_b = seg_query_b;  rec.query_lat_e = query_lat_cy.size();
      rec.insert_lat_b = seg_insert_b; rec.insert_lat_e = insert_lat_cy.size();
      rec.delete_lat_b = seg_delete_b; rec.delete_lat_e = delete_lat_cy.size();
      segment_recs.push_back(std::move(rec));
    }

    if (warmup_check_mode) {
      std::ostringstream hdr;
      hdr << "Algorithm=" << index_type << ";"
          << "Data=" << std::filesystem::path(keys_file_path).filename().string() << ";"
          << config_kv_string();
      const std::string curve_path =
          out_path + ".warmup." + index_type + ".seed-" + std::to_string(random_seed) + ".svg";
      std::vector<std::pair<uint64_t, uint64_t>> pts_ns;
      pts_ns.reserve(total_lat_pts.size());
      for (const auto &p : total_lat_pts) {
        pts_ns.emplace_back(p.first, static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(p.second))));
      }
      write_latency_curve_svg(curve_path, pts_ns, hdr.str());
      delete index;
      return;
    }

    uint64_t totalCycles = searchCycles + insertCycles + deleteCycles;

    const std::string data_name = std::filesystem::path(keys_file_path).filename().string();
    const uint64_t t0 = first_ts_;
    const uint64_t start_time = t0 + static_cast<uint64_t>(time_window);
    const long updateLength = (t_last > start_time)
                                  ? static_cast<long>((t_last - start_time) / static_cast<uint64_t>(time_window))
                                  : 0L;

    std::ostringstream line;
    const long long mem_bytes_end = index->memory_consumption();
    line << "Algorithm=" << index_type
         << ";Data=" << data_name
         << ";TimeWindow=" << time_window
         << ";ScanNum=" << scan_num
         << ";UpdateLength=" << updateLength
         << ";Memory=" << mem_bytes_end
         << ";WallTime=" << std::fixed << std::setprecision(6) << wall_s << ";"
         << ";WallTimeMeasured=" << std::fixed << std::setprecision(6) << wall_meas_s << ";"
         << std::defaultfloat;
    line << config_kv_string();
    if (stream_mode == "scan") {
      line << "ScanTuples=" << scanTuples << ";";
    }

    auto stats_cycles_to_ns = [&](const LatencyStats &cy) -> LatencyStats {
      LatencyStats ns{};
      ns.n = cy.n;
      if (cy.n == 0) return ns;
      ns.mn = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.mn)));
      ns.p50 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p50)));
      ns.p90 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p90)));
      ns.p99 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p99)));
      ns.p999 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p999)));
      ns.p9999 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p9999)));
      ns.mx = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.mx)));
      ns.avg = static_cast<double>(tn.tsc2ns_delta(static_cast<int64_t>(std::llround(cy.avg))));
      return ns;
    };

    if (latency_sample) {
      auto st_total = stats_cycles_to_ns(summarize_latencies(total_lat_cy, "total", std::cerr));
      auto st_q = stats_cycles_to_ns(summarize_latencies(query_lat_cy, "query", std::cerr));
      auto st_i = stats_cycles_to_ns(summarize_latencies(insert_lat_cy, "insert", std::cerr));
      auto st_d = stats_cycles_to_ns(summarize_latencies(delete_lat_cy, "delete", std::cerr));

      emit_latency_stats_kv(line, "total", st_total);
      emit_latency_stats_kv(line, "query", st_q);
      emit_latency_stats_kv(line, "insert", st_i);
      emit_latency_stats_kv(line, "delete", st_d);
    } else {
      line << "SearchTime=" << (double)tn.tsc2ns_delta(static_cast<int64_t>(searchCycles)) / 1e9
           << ";InsertTime=" << (double)tn.tsc2ns_delta(static_cast<int64_t>(insertCycles)) / 1e9
           << ";DeleteTime=" << (double)tn.tsc2ns_delta(static_cast<int64_t>(deleteCycles)) / 1e9
           << ";TotalTime=" << (double)tn.tsc2ns_delta(static_cast<int64_t>(totalCycles)) / 1e9
           << ";TotalCount=" << totalCount << ";";
    }

    if (!cuts.empty()) {
      std::ofstream ofile(out_path, std::ios::app);
      for (const auto &rec : segment_recs) {
        const double seg_wall_s =
            (double)tn.tsc2ns_delta(rec.seg_c1 - rec.seg_c0) / 1e9;
        const double seg_wall_meas_s =
            (!rec.meas_valid) ? 0.0 : (double)tn.tsc2ns_delta(rec.seg_meas_c1 - rec.seg_meas_c0) / 1e9;
        ofile << format_segment_stats_line(
            rec.seg_idx,
            rec.seg_start,
            rec.seg_end,
            (segment_by == "op"),
            rec.mem_bytes,
            seg_wall_s,
            seg_wall_meas_s,
            rec.is_scan_mode,
            rec.scanTuples,
            rec.totalCount, rec.searchCycles, rec.insertCycles, rec.deleteCycles,
            rec.searchOps, rec.insertOps, rec.deleteOps,
            total_lat_cy, rec.total_lat_b, rec.total_lat_e,
            query_lat_cy, rec.query_lat_b, rec.query_lat_e,
            insert_lat_cy, rec.insert_lat_b, rec.insert_lat_e,
            delete_lat_cy, rec.delete_lat_b, rec.delete_lat_e,
            tn,
            index_type,
            keys_file_path,
            time_window,
            scan_num,
            latency_sample,
            config_kv_string())
              << std::endl;
      }
      ofile.close();
    } else {
      std::ofstream ofile(out_path, std::ios::app);
      ofile << line.str() << std::endl;
      ofile.close();
    }

    delete index;
  }

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

  static constexpr uint64_t kPreMagic = 0x4752455F50524538ULL;

  static bool path_looks_like_dir(const std::string &p) {
    if (p.empty()) return false;
    const char last = p.back();
    return last == '/' || last == '\\';
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
      if (!pre_run && !std::filesystem::exists(p)) {
        const auto legacy_dir = derived_pre_run_dir_path_with_hash_(output_path, pre_run_sig_hash_legacy_());
        const auto legacy = legacy_dir / "pre_run.bin";
        if (std::filesystem::exists(legacy)) return legacy;
      }
      return p;
    }
    if (path_looks_like_dir(pre_run_path)) return pre_run_dir_path() / "pre_run.bin";
    return std::filesystem::path(pre_run_path);
  }

  std::filesystem::path pre_run_config_path() const {
    if (pre_run_path.empty() || path_looks_like_dir(pre_run_path)) {
      return pre_run_dir_path() / "config.txt";
    }
    const std::filesystem::path p(pre_run_path);
    const std::string fname = p.stem().string() + ".config.txt";
    return p.parent_path() / fname;
  }

  std::string pre_run_file_path() const {
    return pre_run_bin_path().string();
  }

  void write_pre_run_config_file(const Precomputed &p, const std::filesystem::path &path) const {
    std::error_code ec;
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    if (ec) {
      COUT_N_EXIT("Failed to create directories for pre_run config output");
    }
    std::ofstream out(path.string(), std::ios::out | std::ios::trunc);
    if (!out){
       COUT_N_EXIT("Failed to open pre_run config file for write");
    }

    const std::time_t now = std::time(nullptr);
    std::tm tm_now{};
#if defined(_WIN32)
    localtime_s(&tm_now, &now);
#else
    tm_now = *std::localtime(&now);
#endif

    out << "GRE pre_run config\n";
    out << "GeneratedAtLocal=" << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << "\n";
    out << "BuildDateTime=" << __DATE__ << " " << __TIME__ << "\n";
    out << "Magic=GRE_PRE8\n";
    out << "PreRunBin=" << pre_run_bin_path().string() << "\n";
    out << "PreRunDir=" << pre_run_dir_path().string() << "\n";
    out << "SigHashHex=" << hex_u64(pre_run_sig_hash()) << "\n";
    out << "KeysFile=" << keys_file_path << "\n";
    out << "TsFile=" << (ts_file_path.empty() ? std::string("(generated)") : ts_file_path) << "\n";
    out << "KeysFileType=" << keys_file_type << "\n";
    out << "TsFileType=" << ts_file_type << "\n";
    out << "TableSize=" << table_size << "\n";
    out << "TimeWindow=" << time_window << "\n";
    out << "StreamMode=" << stream_mode << "\n";
    out << "ScanNum=" << scan_num << "\n";
    out << "Seed=" << random_seed << "\n";
    out << "RWRatio=" << std::setprecision(17) << rw_ratio << "\n";
    out << "QueryHitRatio=" << std::setprecision(17) << query_hit_ratio << "\n";
    out << "QueryDistribution=" << query_distribution << "\n";
    out << "ZipfGenerator=" << zipf_generator << "\n";
    out << "ShuffleKeys=" << (shuffle_keys ? 1 : 0) << "\n";
    out << "HalfRange=" << (half_range ? 1 : 0) << "\n";
    out << "StrongerNonDuplicate=" << (stronger_non_duplicate ? 1 : 0) << "\n";
    if (stronger_non_duplicate) {
      out << "StrongerNonDuplicateDropped=" << stronger_non_duplicate_dropped_ << "\n";
      out << "StrongerNonDuplicateRemaining=" << stronger_non_duplicate_remaining_ << "\n";
    }
    out << "LatencySample=" << (latency_sample ? 1 : 0) << "\n";
    out << "LatencySampleRatio=" << std::setprecision(17) << latency_sample_ratio << "\n";
    out << "WarmUp=" << warm_up << "\n";
    out << "SegmentFile=" << segment_file_path << "\n";
    out << "KeyBytes=" << sizeof(KEY_TYPE) << "\n";
    out << "PayloadBytes=" << sizeof(PAYLOAD_TYPE) << "\n";
    out << "InsertN=" << p.insert_ops.size() << "\n";
    out << "QueryN=" << p.query_ops.size() << "\n";
    out << "InitEnd=" << p.init_end << "\n";

    out.close();
  }

  void load_inputs() {
    if (keys_file_type == "binary") {
      n_ = load_binary_data(keys_, table_size, keys_file_path);
    } else if (keys_file_type == "text") {
      n_ = load_text_data(keys_, table_size, keys_file_path);
    } else {
      COUT_N_EXIT("Unsupported keys_file_type (expected binary/text)");
    }
    std::cout << "[streaming] load_inputs: n_=" << n_ << " keys_file=" << keys_file_path << std::endl;
    if (n_ <= 0) {
      COUT_N_EXIT("Failed to load keys_file");
    }

    long long ts_size = -1;
    if (ts_file_path.empty()) {
      ts_ = new PAYLOAD_TYPE[static_cast<size_t>(n_)];
      for (long long i = 0; i < n_; ++i) {
        ts_[static_cast<size_t>(i)] = static_cast<PAYLOAD_TYPE>(i + 1);
      }
      ts_size = n_;
    } else {
      if (ts_file_type == "binary") {
        ts_size = load_binary_data(ts_, n_, ts_file_path);
      } else if (ts_file_type == "text") {
        ts_size = load_text_data(ts_, n_, ts_file_path);
      } else {
        COUT_N_EXIT("Unsupported ts_file_type (expected binary/text)");
      }
      if (ts_size != n_) {
        const long long new_n = std::min<long long>(n_, ts_size);
        std::cerr << "[streaming] warning: keys and ts lengths differ; truncating to min."
                  << " keys_n=" << n_
                  << " ts_n=" << ts_size
                  << " new_n=" << new_n
                  << std::endl;
        n_ = new_n;
        if (n_ <= 0) {
          COUT_N_EXIT("[streaming] after truncation, dataset is empty (n_<=0)");
        }
      }
    }

    if (shuffle_keys) {
      std::mt19937_64 rng(static_cast<uint64_t>(random_seed));
      std::shuffle(keys_, keys_ + static_cast<size_t>(n_), rng);
    }

    stronger_non_duplicate_dropped_ = 0;
    stronger_non_duplicate_remaining_ = static_cast<size_t>(n_);

    if (half_range) {
      if constexpr (std::is_same<KEY_TYPE, uint64_t>::value) {
        for (size_t i = 0; i < static_cast<size_t>(n_); ++i) {
          keys_[i] = static_cast<KEY_TYPE>(static_cast<uint64_t>(keys_[i]) / 2ULL);
        }
      } else {
        COUT_N_EXIT("--half_range requires KEY_TYPE=uint64_t");
      }
    }

    if (stronger_non_duplicate) {
      if constexpr (!std::is_integral<KEY_TYPE>::value) {
        COUT_N_EXIT("--stronger_non_duplicate requires integral KEY_TYPE");
      } else {
        const size_t n0 = static_cast<size_t>(n_);
        std::vector<std::pair<KEY_TYPE, size_t>> sorted;
        sorted.reserve(n0);
        for (size_t i = 0; i < n0; ++i) sorted.emplace_back(keys_[i], i);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto &a, const auto &b) {
                    if (a.first < b.first) return true;
                    if (b.first < a.first) return false;
                    return a.second < b.second;
                  });

        std::vector<uint8_t> drop(n0, 0);
        bool has_last = false;
        KEY_TYPE last_kept{};
        for (const auto &kv : sorted) {
          const KEY_TYPE k = kv.first;
          const size_t idx = kv.second;
          if (!has_last) {
            has_last = true;
            last_kept = k;
            continue;
          }
          const uint64_t ku = static_cast<uint64_t>(k);
          const uint64_t lu = static_cast<uint64_t>(last_kept);
          const bool adjacent_or_equal =
              (ku == lu) || (lu != std::numeric_limits<uint64_t>::max() && ku == lu + 1ULL);
          if (adjacent_or_equal) {
            drop[idx] = 1;
            continue;
          }
          last_kept = k;
        }

        size_t w = 0;
        size_t dropped = 0;
        for (size_t i = 0; i < n0; ++i) {
          if (drop[i]) {
            ++dropped;
            continue;
          }
          keys_[w] = keys_[i];
          ts_[w] = ts_[i];
          ++w;
        }
        stronger_non_duplicate_dropped_ = dropped;
        stronger_non_duplicate_remaining_ = w;
        n_ = static_cast<long long>(w);

        if (ts_file_path.empty()) {
          for (size_t i = 0; i < static_cast<size_t>(n_); ++i) {
            ts_[i] = static_cast<PAYLOAD_TYPE>(i + 1);
          }
        }

        std::cout << "[streaming] stronger_non_duplicate: dropped=" << stronger_non_duplicate_dropped_
                  << " remaining=" << stronger_non_duplicate_remaining_
                  << " (n0=" << n0 << ")" << std::endl;
      }
    }

    const size_t init_end = initial_end_idx();
    INVARIANT(static_cast<uint64_t>(ts_[0]) > 0);
    if (init_end >= static_cast<size_t>(n_)) {
      std::ostringstream oss;
      oss << "After pre-run transforms, dataset is too small for the configured time_window: "
          << "init_end=" << init_end << " n_=" << n_ << " time_window=" << time_window
          << " (note: --stronger_non-duplicate/--half_range may shrink the dataset; "
          << "reduce --time_window, increase --table_size, or disable the transforms)";
      COUT_N_EXIT(oss.str().c_str());
    }
  }

  size_t initial_end_idx() const {
    INVARIANT(n_ > 0);
    const uint64_t t0 = static_cast<uint64_t>(ts_[0]);
    const uint64_t start_time = t0 + static_cast<uint64_t>(time_window);
    size_t i = 0;
    while (i < static_cast<size_t>(n_) && static_cast<uint64_t>(ts_[i]) < start_time) {
      ++i;
    }
    INVARIANT(i > 0);
    return i;
  }

  void cleanup_inputs() {
    delete[] keys_;
    delete[] ts_;
    keys_ = nullptr;
    ts_ = nullptr;
    n_ = 0;
    first_ts_ = 0;
  }

  static std::vector<uint64_t> read_segment_cuts(const std::string &path) {
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

  static std::string format_segment_stats_line(size_t seg_idx,
                                               uint64_t seg_start,
                                               uint64_t seg_end,
                                               bool segment_by_op,
                                               long long memory_bytes,
                                               double seg_wall_s,
                                               double seg_wall_meas_s,
                                               bool is_scan_mode,
                                               uint64_t scan_tuples,
                                               uint64_t totalCount,
                                               uint64_t searchCycles,
                                               uint64_t insertCycles,
                                               uint64_t deleteCycles,
                                               uint64_t searchOps,
                                               uint64_t insertOps,
                                               uint64_t deleteOps,
                                               const std::vector<uint64_t> &total_lat_cy,
                                               size_t total_lat_b,
                                               size_t total_lat_e,
                                               const std::vector<uint64_t> &query_lat_cy,
                                               size_t query_lat_b,
                                               size_t query_lat_e,
                                               const std::vector<uint64_t> &insert_lat_cy,
                                               size_t insert_lat_b,
                                               size_t insert_lat_e,
                                               const std::vector<uint64_t> &delete_lat_cy,
                                               size_t delete_lat_b,
                                               size_t delete_lat_e,
                                               TSCNS &tn,
                                               const std::string &index_type,
                                               const std::string &keys_file_path,
                                               size_t time_window,
                                               size_t scan_num,
                                               bool latency_sample,
                                               const std::string &cfg_kv) {
    const std::string data_name = std::filesystem::path(keys_file_path).filename().string();

    std::ostringstream line;
    line << "Algorithm=" << index_type
         << ";Data=" << data_name
         << ";TimeWindow=" << time_window
         << ";ScanNum=" << scan_num
         << ";SegmentIdx=" << seg_idx
         << (segment_by_op ? ";SegmentStartOp=" : ";SegmentStart=") << seg_start
         << (segment_by_op ? ";SegmentEndOp=" : ";SegmentEnd=") << seg_end
         << ";Memory=" << memory_bytes
         << ";SegmentWallTime=" << std::fixed << std::setprecision(6) << seg_wall_s << ";"
         << ";SegmentWallTimeMeasured=" << std::fixed << std::setprecision(6) << seg_wall_meas_s << ";"
         << std::defaultfloat;
    line << cfg_kv;
    if (is_scan_mode) {
      line << "ScanTuples=" << scan_tuples << ";";
    }

    if (latency_sample) {
      auto slice = [&](const std::vector<uint64_t> &v, size_t b, size_t e) -> std::vector<uint64_t> {
        if (b >= e || b >= v.size()) return {};
        e = std::min(e, v.size());
        return std::vector<uint64_t>(v.begin() + static_cast<std::ptrdiff_t>(b),
                                     v.begin() + static_cast<std::ptrdiff_t>(e));
      };
      auto stats_cycles_to_ns = [&](const LatencyStats &cy) -> LatencyStats {
        LatencyStats ns{};
        ns.n = cy.n;
        if (cy.n == 0) return ns;
        ns.mn = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.mn)));
        ns.p50 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p50)));
        ns.p90 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p90)));
        ns.p99 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p99)));
        ns.p999 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p999)));
        ns.p9999 = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.p9999)));
        ns.mx = static_cast<uint64_t>(tn.tsc2ns_delta(static_cast<int64_t>(cy.mx)));
        ns.avg = static_cast<double>(tn.tsc2ns_delta(static_cast<int64_t>(std::llround(cy.avg))));
        return ns;
      };

      auto st_total = stats_cycles_to_ns(summarize_latencies(slice(total_lat_cy, total_lat_b, total_lat_e), "total", std::cerr));
      auto st_q = stats_cycles_to_ns(summarize_latencies(slice(query_lat_cy, query_lat_b, query_lat_e), "query", std::cerr));
      auto st_i = stats_cycles_to_ns(summarize_latencies(slice(insert_lat_cy, insert_lat_b, insert_lat_e), "insert", std::cerr));
      auto st_d = stats_cycles_to_ns(summarize_latencies(slice(delete_lat_cy, delete_lat_b, delete_lat_e), "delete", std::cerr));

      emit_latency_stats_kv(line, "total", st_total);
      emit_latency_stats_kv(line, "query", st_q);
      emit_latency_stats_kv(line, "insert", st_i);
      emit_latency_stats_kv(line, "delete", st_d);

      line << "SearchOps=" << searchOps << ";";
      line << "InsertOps=" << insertOps << ";";
      line << "DeleteOps=" << deleteOps << ";";
    } else {
      const uint64_t totalCycles = searchCycles + insertCycles + deleteCycles;
      const double search_s = (double)tn.tsc2ns_delta(static_cast<int64_t>(searchCycles)) / 1e9;
      const double insert_s = (double)tn.tsc2ns_delta(static_cast<int64_t>(insertCycles)) / 1e9;
      const double delete_s = (double)tn.tsc2ns_delta(static_cast<int64_t>(deleteCycles)) / 1e9;
      const double total_s = (double)tn.tsc2ns_delta(static_cast<int64_t>(totalCycles)) / 1e9;
      const uint64_t totalOps = searchOps + insertOps + deleteOps;

      line << "SearchTime=" << search_s
           << ";InsertTime=" << insert_s
           << ";DeleteTime=" << delete_s
           << ";TotalTime=" << total_s
           << ";TotalCount=" << totalCount
           << ";SearchOps=" << searchOps
           << ";InsertOps=" << insertOps
           << ";DeleteOps=" << deleteOps
           << ";TotalOps=" << totalOps;
      if (total_s > 0) line << ";Throughput=" << (static_cast<double>(totalOps) / total_s);
      line << ";";
    }
    return line.str();
  }

  static uint8_t periodic_bit(uint64_t idx1, uint64_t interval) {
    if (interval == 0) return 0;
    return ((idx1 % interval) == 0) ? 1 : 0;
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
      op.type = 0;
      p.ops.push_back(op);
    }

    for (const auto &q : p.query_ops) {
      ReplayOp op;
      op.ts = q.ts;
      op.key = q.key;
      op.aux = q.end_key;
      op.type = 1;
      p.ops.push_back(op);
    }

    for (size_t i = init_end; i < n; ++i) {
      ReplayOp op;
      op.ts = p.insert_ops[i].ts;
      op.key = p.insert_ops[i].key;
      op.aux = static_cast<KEY_TYPE>(0);
      op.type = 2;
      p.ops.push_back(op);
    }

    std::sort(p.ops.begin(), p.ops.end(),
              [](const ReplayOp &a, const ReplayOp &b) {
                if (a.ts != b.ts) return a.ts < b.ts;
                return a.type < b.type;
              });
  }

  Precomputed precompute_sequences() {
    Precomputed p;
    p.n = static_cast<uint64_t>(n_);
    p.time_window = static_cast<uint64_t>(time_window);
    p.init_end = static_cast<uint64_t>(initial_end_idx());

    const size_t init_end = static_cast<size_t>(p.init_end);
    const uint64_t t0 = static_cast<uint64_t>(ts_[0]);
    const uint64_t start_time = t0 + static_cast<uint64_t>(time_window);

    std::mt19937_64 rng(random_seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    const uint64_t zipf_domain = std::max<uint64_t>(1, p.init_end);
    UnscrambledZipfianGenerator zipf_gen(zipf_domain, random_seed);

    size_t scrambled_seed = random_seed;
    ScrambledZipfianGenerator scrambled_zipf_gen(
        static_cast<int>(std::min<uint64_t>(zipf_domain, static_cast<uint64_t>(std::numeric_limits<int>::max()))),
        &scrambled_seed);

    const bool need_scan_range = (stream_mode == "scan");
    BTreeInterface<KEY_TYPE, PAYLOAD_TYPE> btree;
    if (need_scan_range) {
      std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> init;
      init.reserve(static_cast<size_t>(p.init_end));
      for (size_t i = 0; i < static_cast<size_t>(p.init_end); ++i) {
        init.emplace_back(keys_[i], static_cast<PAYLOAD_TYPE>(0));
      }
      std::sort(init.begin(), init.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      btree.bulk_load(init.data(), init.size(), nullptr);
    }

    p.insert_ops.clear();
    p.query_ops.clear();
    p.insert_ops.reserve(static_cast<size_t>(n_));
    for (size_t i = 0; i < static_cast<size_t>(n_); ++i) {
      InsertOp ins;
      ins.key = keys_[i];
      ins.ts = static_cast<uint64_t>(ts_[i]);
      p.insert_ops.push_back(ins);
    }
    const size_t ins_cnt = (static_cast<size_t>(n_) > init_end) ? (static_cast<size_t>(n_) - init_end) : 0;
    const size_t approx_q_cnt = static_cast<size_t>(std::ceil(std::max<double>(0.0, rw_ratio) * ins_cnt));
    p.query_ops.reserve(approx_q_cnt);

    size_t delete_ptr = 0;
    if (need_scan_range) {
      // Ensure btree contains the same set as prepare_index bulk-load: initial window keys.
      // (Already bulk-loaded above.)
    }

    const double rw = std::max<double>(0.0, rw_ratio);
    const size_t q_floor = static_cast<size_t>(std::floor(rw));
    const double q_frac = rw - static_cast<double>(q_floor);
    double q_carry = 0.0;
    for (size_t i = init_end; i < static_cast<size_t>(n_); ++i) {
      const uint64_t t_ins = static_cast<uint64_t>(ts_[i]);
      const uint64_t t_prev = static_cast<uint64_t>(ts_[i - 1]);
      const uint64_t t_low = std::max<uint64_t>(t_prev, start_time);

      size_t qn = q_floor;
      q_carry += q_frac;
      if (q_carry >= 1.0) {
        ++qn;
        q_carry -= 1.0;
      }

      std::vector<uint64_t> q_ts_list;
      q_ts_list.reserve(qn);
      for (size_t qi = 0; qi < qn; ++qi) {
        uint64_t q_ts = t_ins;
        if (t_ins > t_low) {
          const uint64_t span = t_ins - t_low;
          q_ts = (t_low + 1ULL) + (rng() % span);
        }
        q_ts_list.push_back(q_ts);
      }
      std::sort(q_ts_list.begin(), q_ts_list.end());

      for (const uint64_t q_ts : q_ts_list) {
        const uint64_t threshold =
            (q_ts > static_cast<uint64_t>(time_window)) ? (q_ts - static_cast<uint64_t>(time_window)) : 0ULL;
        while (delete_ptr < i && static_cast<uint64_t>(ts_[delete_ptr]) < threshold) {
          if (need_scan_range) {
            (void)btree.remove(keys_[delete_ptr], nullptr);
          }
          ++delete_ptr;
        }
        const size_t lo = delete_ptr;
        const size_t hi = (i > 0) ? (i - 1) : 0;
        size_t pick = i;
        const bool want_hit = (uni01(rng) < query_hit_ratio);
        if (want_hit) {
          if (hi >= lo) {
            const uint64_t span = static_cast<uint64_t>(hi - lo + 1);
            if (query_distribution == "uniform") {
              pick = lo + static_cast<size_t>(rng() % span);
            } else {
              if (zipf_generator == "scrambled") {
                uint64_t off = static_cast<uint64_t>(scrambled_zipf_gen.nextValue());
                while (off >= span) off = static_cast<uint64_t>(scrambled_zipf_gen.nextValue());
                pick = lo + static_cast<size_t>(off);
              } else {
                uint64_t r = zipf_gen.nextValue();
                while (r >= span) r = zipf_gen.nextValue();
                pick = hi - static_cast<size_t>(r);
              }
            }
          }
        } else {
          const size_t win_k = (hi >= lo) ? (hi - lo + 1) : 0;
          const size_t n_sz = static_cast<size_t>(n_);

          const size_t future_len = (i < n_sz && win_k > 0) ? std::min(win_k, n_sz - i) : 0;

          const size_t need_past = (future_len < win_k) ? (win_k - future_len) : 0;
          const size_t past_len = (need_past > 0) ? std::min(need_past, lo) : 0;

          const size_t combined = future_len + past_len;
          if (combined > 0) {
            const size_t r = static_cast<size_t>(rng() % static_cast<uint64_t>(combined));
            if (r < future_len) {
              pick = i + r;
            } else {
              const size_t past_lo = lo - past_len;
              pick = past_lo + (r - future_len);
            }
          } else {
            pick = i;
          }
        }

        QueryOp q;
        q.key = keys_[pick];
        q.ts = q_ts;
        q.end_key = static_cast<KEY_TYPE>(0);
        if (need_scan_range) {
          KEY_TYPE end_key = static_cast<KEY_TYPE>(0);
          (void)btree.nth_key_or_last_from(q.key, scan_num, end_key);
          q.end_key = end_key;
        }
        p.query_ops.push_back(q);
      }

      if (need_scan_range) {
        (void)btree.put(keys_[i], static_cast<PAYLOAD_TYPE>(0), nullptr);
      }
    }

    return p;
  }

  void write_pre_run_file(const Precomputed &p, const std::string &path) const {
    std::error_code ec;
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    if (ec) {
      COUT_N_EXIT("Failed to create directories for pre_run output");
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      COUT_N_EXIT("Failed to open pre_run_path for write");
    }

    const uint64_t magic = kPreMagic;
    const uint64_t insert_n = static_cast<uint64_t>(p.insert_ops.size());
    const uint64_t query_n = static_cast<uint64_t>(p.query_ops.size());
    const uint64_t scan_num_u64 = static_cast<uint64_t>(scan_num);
    const uint64_t seed_u64 = static_cast<uint64_t>(random_seed);
    const double rw_ratio_f64 = rw_ratio;
    const double hit_ratio_f64 = query_hit_ratio;
    const double latency_ratio_f64 = latency_sample_ratio;
    const uint64_t flags =
        (stream_mode == "scan" ? 1ULL : 0ULL) |
        (query_distribution == "zipf" ? 2ULL : 0ULL) |
        (zipf_generator == "scrambled" ? 4ULL : 0ULL) |
        (shuffle_keys ? 8ULL : 0ULL);
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&p.n), sizeof(p.n));
    out.write(reinterpret_cast<const char*>(&p.time_window), sizeof(p.time_window));
    out.write(reinterpret_cast<const char*>(&p.init_end), sizeof(p.init_end));
    out.write(reinterpret_cast<const char*>(&scan_num_u64), sizeof(scan_num_u64));
    out.write(reinterpret_cast<const char*>(&seed_u64), sizeof(seed_u64));
    out.write(reinterpret_cast<const char*>(&rw_ratio_f64), sizeof(rw_ratio_f64));
    out.write(reinterpret_cast<const char*>(&hit_ratio_f64), sizeof(hit_ratio_f64));
    out.write(reinterpret_cast<const char*>(&latency_ratio_f64), sizeof(latency_ratio_f64));
    out.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    out.write(reinterpret_cast<const char*>(&insert_n), sizeof(insert_n));
    out.write(reinterpret_cast<const char*>(&query_n), sizeof(query_n));

    for (const auto &ins : p.insert_ops) {
      out.write(reinterpret_cast<const char*>(&ins.key), sizeof(KEY_TYPE));
      out.write(reinterpret_cast<const char*>(&ins.ts), sizeof(ins.ts));
    }
    for (const auto &q : p.query_ops) {
      out.write(reinterpret_cast<const char*>(&q.key), sizeof(KEY_TYPE));
      out.write(reinterpret_cast<const char*>(&q.end_key), sizeof(KEY_TYPE));
      out.write(reinterpret_cast<const char*>(&q.ts), sizeof(q.ts));
    }
    out.close();
  }

  Precomputed read_pre_run_file(const std::string &path) const {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      COUT_N_EXIT("Failed to open pre_run_path for read (run pre_run first)");
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
    if (std::isfinite(latency_ratio_f64) && std::isfinite(latency_sample_ratio) &&
        latency_ratio_f64 != latency_sample_ratio) {
      std::cerr << "[streaming] note: pre_run was generated with latency_sample_ratio="
                << std::setprecision(17) << latency_ratio_f64
                << " but current run uses latency_sample_ratio="
                << std::setprecision(17) << latency_sample_ratio
                << " (this is OK; sampling does not affect the op stream)"
                << std::endl;
    }
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

  index_t* prepare_index() {
    Param param(1, 0);

    index_t *index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);
    index->init(&param);
    index->bulk_load(init_kv_sorted_.data(), init_kv_sorted_.size(), &param);
    if (keys_ || ts_) {
      delete[] keys_;
      delete[] ts_;
      keys_ = nullptr;
      ts_ = nullptr;
    }
    return index;
  }
};