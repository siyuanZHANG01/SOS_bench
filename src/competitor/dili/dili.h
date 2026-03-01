#pragma once

// DILI upstream defines KEY_TYPE / PAYLOAD_TYPE as macros in global_typedef.h.
// Those macros break GRE's templated interfaces (template<class KEY_TYPE, class PAYLOAD_TYPE>).
// Keep behavior impact minimal by restoring any pre-existing macro definitions after include.
#ifdef KEY_TYPE
#pragma push_macro("KEY_TYPE")
#undef KEY_TYPE
#define GRE_RESTORE_KEY_TYPE 1
#endif
#ifdef PAYLOAD_TYPE
#pragma push_macro("PAYLOAD_TYPE")
#undef PAYLOAD_TYPE
#define GRE_RESTORE_PAYLOAD_TYPE 1
#endif
#include "./src/src/dili/DILI.h"
#ifdef GRE_RESTORE_KEY_TYPE
#pragma pop_macro("KEY_TYPE")
#undef GRE_RESTORE_KEY_TYPE
#else
#ifdef KEY_TYPE
#undef KEY_TYPE
#endif
#endif
#ifdef GRE_RESTORE_PAYLOAD_TYPE
#pragma pop_macro("PAYLOAD_TYPE")
#undef GRE_RESTORE_PAYLOAD_TYPE
#else
#ifdef PAYLOAD_TYPE
#undef PAYLOAD_TYPE
#endif
#endif
#include "../indexInterface.h"

#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

template<class KEY_TYPE, class PAYLOAD_TYPE>
class diliInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) override {}

  void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) override;

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override;

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override;

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override;

  bool remove(KEY_TYPE key, Param *param = nullptr) override;

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr) override;

  long long memory_consumption() override { return static_cast<long long>(index.total_size_logical_precise()); }

private:
  DILI index;
  bool mirror_dir_inited = false;

  static_assert(std::is_integral<KEY_TYPE>::value, "diliInterface requires integral KEY_TYPE");
  static_assert(std::is_integral<PAYLOAD_TYPE>::value, "diliInterface requires integral PAYLOAD_TYPE");
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void diliInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                      Param *param) {
  // DILI's bulk_load builds/restores a BU-Tree "mirror" on disk via mirror_dir.
  // If mirror_dir is left empty, it may try to read/write in the current working directory.
  const std::string mirror_dir = "src/competitor/dili/src/build/data/buTree";
  if (!mirror_dir_inited) {
    index.set_mirror_dir(mirror_dir);
    mirror_dir_inited = true;
  }

  // Align with the "always rebuild mirror" behavior:
  // remove the mirror directory and recreate it on every bulk_load.
  {
    std::error_code ec;
    std::filesystem::remove_all(mirror_dir, ec);
    // (Re)create even if remove_all failed or directory didn't exist.
    std::filesystem::create_directories(mirror_dir, ec);
  }

  // Avoid an extra copy: call the keyArray/recordPtrArray overload directly.
  // Note: DILI expects an extra sentinel key at the end.
  keyArray keys = std::make_unique<keyType []>(num + 1);
  recordPtrArray ptrs = std::make_unique<recordPtr []>(num + 1);
  for (size_t i = 0; i < num; ++i) {
    keys[i] = static_cast<keyType>(key_value[i].first);
    ptrs[i] = static_cast<recordPtr>(key_value[i].second);
  }

  // Debug: print key count and validate input order before building DILI.
  std::cerr << "[DILI][bulk_load] n_keys=" << num;
  if (num > 0) std::cerr << " first=" << keys[0] << " last=" << keys[num - 1];
  std::cerr << std::endl;
  for (size_t i = 1; i < num; ++i) {
    if (keys[i] <= keys[i - 1]) {
      std::cerr << "[DILI][bulk_load][fatal] keys must be strictly increasing, but keys[" << (i - 1)
                << "]=" << keys[i - 1] << " keys[" << i << "]=" << keys[i]
                << " (duplicate or unsorted input). Aborting before build." << std::endl;
      std::abort();
    }
  }
  // For correctness, DILI expects keys to be sorted (monotonic) for mirror construction and build.
  // microbench already sorts init_keys before bulk_load, but keep the assumption explicit.
  // Sentinel key: must be strictly greater than the last key.
  keys[num] = (num > 0) ? (keys[num - 1] + 1) : 0;
  ptrs[num] = -1;

  index.bulk_load(keys, ptrs, static_cast<long>(num));

  // Post-build correctness check (sampling): check every 10,000th key.
  // This is meant to catch obvious build corruption early without the cost of full validation.
  static constexpr size_t kValidateStride = 100000;
  std::cerr << "[DILI][validate] enabled stride=" << kValidateStride << " (n_keys=" << num << ")\n";
  for (size_t i = 0; i < num; i += kValidateStride) {
    const keyType k = keys[i];
    const recordPtr expected = ptrs[i];
    const recordPtr got = index.search(k);
    std::cout<< got << expected << std::endl;
    if (got != expected) {
      std::cerr << "[DILI][validate][fatal] mismatch at i=" << i
                << " key=" << k
                << " expected=" << expected
                << " got=" << got
                << " stride=" << kValidateStride
                << " (aborting)\n";
      std::abort();
    }
  }
  std::cerr << "[DILI][validate] OK\n";
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  recordPtr res = index.search(static_cast<keyType>(key));
  if (res >= 0) {
    val = static_cast<PAYLOAD_TYPE>(res);
    return true;
  }
  return false;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return index.insert(static_cast<keyType>(key), static_cast<recordPtr>(value));
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  // DILI doesn't provide in-place update; emulate with erase + insert.
  if (!index.erase(static_cast<keyType>(key))) return false;
  return index.insert(static_cast<keyType>(key), static_cast<recordPtr>(value));
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool diliInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return index.erase(static_cast<keyType>(key));
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t diliInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
  if (key_num == 0) return 0;
  (void)result;
  static thread_local std::vector<recordPtr> ptrs;
  ptrs.resize(key_num);

  if (!param) return 0;
  const keyType k1 = static_cast<keyType>(key_low_bound);
  const keyType ub_incl = static_cast<keyType>(param->scan_end_key);
  const keyType k2_excl =
      (ub_incl == std::numeric_limits<keyType>::max()) ? ub_incl : static_cast<keyType>(ub_incl + 1);
  int got = index.range_query(k1, k2_excl, ptrs.data());
  if (got < 0) return 0;
  return std::min(static_cast<size_t>(got), key_num);
}

