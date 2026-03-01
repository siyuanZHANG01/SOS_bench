#pragma once

#include "./src/src/core/sali.h"
#include "../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class saliInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
  void init(Param *param = nullptr) override {}

  void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) override;

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override;

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override;

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override;

  bool remove(KEY_TYPE key, Param *param = nullptr) override;

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
              Param *param = nullptr) override;

  long long memory_consumption() override { return static_cast<long long>(index.total_size()); }

private:
  sali::SALI<KEY_TYPE, PAYLOAD_TYPE> index;
};

template<class KEY_TYPE, class PAYLOAD_TYPE>
void saliInterface<KEY_TYPE, PAYLOAD_TYPE>::bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num,
                                                      Param *param) {
  index.bulk_load(key_value, static_cast<int>(num));
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool saliInterface<KEY_TYPE, PAYLOAD_TYPE>::get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param) {
  return index.at(key, val);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool saliInterface<KEY_TYPE, PAYLOAD_TYPE>::put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  index.insert(key, value);
  return true;
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool saliInterface<KEY_TYPE, PAYLOAD_TYPE>::update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param) {
  return index.update(key, value);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
bool saliInterface<KEY_TYPE, PAYLOAD_TYPE>::remove(KEY_TYPE key, Param *param) {
  return index.remove(key);
}

template<class KEY_TYPE, class PAYLOAD_TYPE>
size_t saliInterface<KEY_TYPE, PAYLOAD_TYPE>::scan(KEY_TYPE key_low_bound, size_t key_num,
                                                   std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                                                   Param *param) {
  if (!result || key_num == 0) return 0;
  int got = index.range_query_len(result, key_low_bound, static_cast<int>(key_num));
  return got > 0 ? static_cast<size_t>(got) : 0;
}
