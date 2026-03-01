#pragma once

// DyTIS defines global `struct Key` / `struct Value` in util/pair.h, which can
// conflict with other competitors (e.g., ART's `class Key`). Avoid touching the
// upstream code by locally renaming those identifiers only while including DyTIS.
#pragma push_macro("Key")
#pragma push_macro("Value")
#undef Key
#undef Value
#define Key DyTIS_Key
#define Value DyTIS_Value
#include "./src/util/pair.h"
#undef Key
#undef Value
#pragma pop_macro("Value")
#pragma pop_macro("Key")
#include "./src/src/DyTIS.h"
#include "./src/src/DyTIS_impl.h"

#include "../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class dytisInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) override {}

  void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) override;

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override;

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override;

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override;

  bool remove(KEY_TYPE key, Param *param = nullptr) override;

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr) override;

  long long memory_consumption() override { return 0; }

private:
  DyTIS index;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void dytisInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                       Param *param) {
  // DyTIS doesn't have a dedicated bulk-load build; do sequential inserts.
  for (size_t i = 0; i < num; ++i) {
    index.Insert(key_value[i].first, key_value[i].second);
  }
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool dytisInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  PAYLOAD_TYPE *p = index.Find(key);
  if (!p) return false;
  val = *p;
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool dytisInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  index.Insert(key, value);
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool dytisInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return index.Update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool dytisInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return index.Delete(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t dytisInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                    std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                    Param *param) {
  (void)result;
  if (key_num == 0) return 0;
  auto vals = index.Scan(key_low_bound, key_num);
  // DyTIS::Scan allocates; free it to avoid leaking in scan-heavy workloads.
  delete[] vals;
  return key_num;
}

