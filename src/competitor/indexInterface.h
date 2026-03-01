#include <cstdint>
#include <iomanip>

#pragma once

struct Param { // for xindex
  size_t worker_num;
  uint32_t thread_id;
  // Optional per-operation context (used by some streaming indexes such as SWIX).
  // In streaming benchmark, this is set to the current event timestamp before each operation.
  uint64_t ts = 0;
  // Optional scan-range context (used by SWIX streaming scan): interpret as upper-bound key for range_search.
  uint64_t scan_end_key = 0;

  Param(size_t worker_num, uint32_t thread_id) : worker_num(worker_num), thread_id(thread_id) {}
};

struct BaseCompare {
  template<class T1, class T2>
  bool operator()(const T1 &x, const T2 &y) const {
    static_assert(
      std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
      "Comparison types must be numeric.");
    return x < y;
  }
};

template<class KEY_TYPE, class PAYLOAD_TYPE, class KeyComparator=BaseCompare>
class indexInterface {
public:
  virtual ~indexInterface() = default;

  virtual void bulk_load(std::pair <KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) = 0;

  virtual bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) = 0;

  virtual bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) = 0;

  virtual bool remove(KEY_TYPE key, Param *param = nullptr) = 0;

  virtual size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                      Param *param = nullptr) = 0;

  virtual void init(Param *param = nullptr) = 0;

  virtual long long memory_consumption() = 0; // bytes
};