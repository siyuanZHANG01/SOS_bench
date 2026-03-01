#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "../indexInterface.h"

// IMTree reference implementation is vendored under the SWIX competitor tree.
// We wrap imtree::IMTree<K, Ts> into GRE's indexInterface.
#include "../swix/src/src/IMTree.hpp"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class imtreeInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    using imtree_t = imtree::IMTree<KEY_TYPE, PAYLOAD_TYPE>;

    imtreeInterface() = default;

    ~imtreeInterface() override {
        delete index_;
        index_ = nullptr;
    }

    void init(Param *param = nullptr) override {
        // Align with SWIX integration: keep streaming competitors single-threaded for now.
        if (param) param->worker_num = 1;
    }

    void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value,
                   size_t num,
                   Param *param = nullptr) override {
        (void)param;
        delete index_;
        index_ = nullptr;

        std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> stream;
        stream.reserve(num);
        for (size_t i = 0; i < num; ++i) stream.push_back(key_value[i]);

        // IMTree requires a merge threshold.
        // Hard-coded per request: 0.15 * 100,000,000 = 15,000,000.
        static constexpr int kMergeThreshold = 15000000;
        index_ = new imtree_t(stream, kMergeThreshold);
    }

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override {
        // IMTree lookup returns a count; expose it via val and return (count > 0).
        const PAYLOAD_TYPE ts = static_cast<PAYLOAD_TYPE>(param->ts);
        std::pair<KEY_TYPE, PAYLOAD_TYPE> q{key, ts};
        KEY_TYPE cnt = 0;
        index_->lookup(q, cnt);
        val = static_cast<PAYLOAD_TYPE>(cnt);
        return cnt != 0;
    }

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        (void)param;
        std::pair<KEY_TYPE, PAYLOAD_TYPE> ins{key, value};
        index_->insert(ins);
        return true;
    }

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        // IMTree doesn't support in-place update.
        (void)key;
        (void)value;
        (void)param;
        return true;
    }

    bool remove(KEY_TYPE key, Param *param = nullptr) override {
        // IMTree evicts by timestamp filtering and periodic merges; treat deletes as no-ops.
        (void)key;
        (void)param;
        return true;
    }

    size_t scan(KEY_TYPE key_low_bound,
                size_t key_num,
                std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr) override {
        (void)result;
        if (key_num == 0) return 0;

        const PAYLOAD_TYPE ts = static_cast<PAYLOAD_TYPE>(param->ts);
        const KEY_TYPE upper = static_cast<KEY_TYPE>(param->scan_end_key);
        std::tuple<KEY_TYPE, PAYLOAD_TYPE, KEY_TYPE> q{key_low_bound, ts, upper};

        thread_local std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> buf;
        buf.clear();
        if (buf.capacity() < key_num) buf.reserve(key_num);
        index_->range_search(q, buf);
        return buf.size();
    }

    long long memory_consumption() override {
        return static_cast<long long>(index_->get_total_size_in_bytes());
    }

private:
    imtree_t *index_ = nullptr;
};

