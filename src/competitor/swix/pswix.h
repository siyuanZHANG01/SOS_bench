#pragma once

#include "./src/src/PSwix_v2.hpp"
#include "../indexInterface.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

// Adapter for PSWIX (parallel SWIX) to the common indexInterface.
//
// Important:
// - PSWIX's intended execution model is "broadcast tasks to all threads, owning thread executes"
//   via SWmeta::within_thread(...). The concurrent replay benchmark implements that dispatcher.
// - This adapter provides accessors to PSWIX's specialized APIs so the benchmark can call them.
//
// The base indexInterface methods are implemented for completeness, but for correct PSWIX semantics
// under concurrent replay you should use within_thread + (lookup/range_query/insert) as in
// src/benchmark/concurrent_evaluation.h.

template<class KEY_TYPE, class PAYLOAD_TYPE>
class pswixInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  using pswix_t = pswix::SWmeta<KEY_TYPE, PAYLOAD_TYPE>;
  using bound_t = pswix::search_bound_type;

  pswixInterface() = default;
  ~pswixInterface() override { delete idx_; }

  void init(Param *param = nullptr) override {
    worker_num_ = param ? param->worker_num : 1;
    pswix::configure_runtime_params(worker_num_, TIME_WINDOW,
                                    NUM_SEARCH_PER_ROUND + NUM_UPDATE_PER_ROUND * 2);
  }

  void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                 Param *param = nullptr) override {
    const size_t wn = param ? param->worker_num : worker_num_;
    // PSWIX bulk-load expects an arrival stream (key, ts) in time order (NOT sorted-by-key).
    std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> stream;
    stream.reserve(num);
    for (size_t i = 0; i < num; ++i) stream.emplace_back(key_value[i].first, key_value[i].second);
    delete idx_;
    idx_ = new pswix_t(static_cast<int>(wn), stream);
  }

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override {
    if (!idx_ || !param) return false;
    bound_t b{};
    if (!idx_->within_thread(param->thread_id, key, b)) return false;
    const int got = idx_->lookup(param->thread_id, key, static_cast<PAYLOAD_TYPE>(param->ts), b);
    val = static_cast<PAYLOAD_TYPE>(got);
    return got > 0;
  }

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
    if (!idx_ || !param) return false;
    bound_t b{};
    if (!idx_->within_thread(param->thread_id, key, b)) return true; // non-owning thread: no-op
    (void)idx_->insert(param->thread_id, key, value, b);
    return true;
  }

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
    return put(key, value, param);
  }

  bool remove(KEY_TYPE key, Param *param = nullptr) override {
    // PSWIX uses time-window semantics; its benchmark does not execute explicit deletes.
    (void)key;
    (void)param;
    return true;
  }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num,
              std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr) override {
    (void)result;
    if (!idx_ || !param || key_num == 0) return 0;
    const KEY_TYPE ub =
        (param->scan_end_key == 0) ? std::numeric_limits<KEY_TYPE>::max()
                                   : static_cast<KEY_TYPE>(param->scan_end_key);
    bound_t b{};
    if (!idx_->within_thread(param->thread_id, key_low_bound, ub, b)) return 0;
    const int got = idx_->range_query(param->thread_id, key_low_bound,
                                      static_cast<PAYLOAD_TYPE>(param->ts), ub, b);
    return got > 0 ? std::min(static_cast<size_t>(got), key_num) : 0;
  }

  long long memory_consumption() override {
    if (!idx_) return 0;
    return static_cast<long long>(idx_->memory_usage());
  }

  // --- PSWIX specialized APIs for the concurrent broadcast dispatcher ---

  pswix_t* raw() { return idx_; }
  const pswix_t* raw() const { return idx_; }

  inline bool within_thread_key(uint32_t thread_id, KEY_TYPE key, bound_t &b) {
    return idx_ && idx_->within_thread(thread_id, key, b);
  }

  inline bool within_thread_range(uint32_t thread_id, KEY_TYPE lo, KEY_TYPE hi, bound_t &b) {
    return idx_ && idx_->within_thread(thread_id, lo, hi, b);
  }

  inline int lookup_raw(uint32_t thread_id, KEY_TYPE key, PAYLOAD_TYPE ts, bound_t &b) {
    return idx_->lookup(thread_id, key, ts, b);
  }

  inline int range_query_raw(uint32_t thread_id, KEY_TYPE lo, PAYLOAD_TYPE ts, KEY_TYPE hi, bound_t &b) {
    return idx_->range_query(thread_id, lo, ts, hi, b);
  }

  inline int insert_raw(uint32_t thread_id, KEY_TYPE key, PAYLOAD_TYPE ts, bound_t &b) {
    return idx_->insert(thread_id, key, ts, b);
  }

private:
  pswix_t *idx_ = nullptr;
  size_t worker_num_ = 1;

  static_assert(std::is_integral<KEY_TYPE>::value, "pswixInterface requires integral KEY_TYPE");
  static_assert(std::is_integral<PAYLOAD_TYPE>::value, "pswixInterface requires integral PAYLOAD_TYPE");
};