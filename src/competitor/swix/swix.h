#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>

#include "../indexInterface.h"

// SWIX reference implementation lives under this submodule directory.
// We wrap swix::SWmeta<K, Ts> into GRE's indexInterface.
#include "./src/src/Swix.hpp"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class swixInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    using swix_t = swix::SWmeta<KEY_TYPE, PAYLOAD_TYPE>;

    swixInterface() = default;

    ~swixInterface() override {
        delete index_;
        index_ = nullptr;
    }

    void init(Param *param = nullptr) override {
        // Temporarily: only single-thread is supported/considered in GRE integration.
        if (param) param->worker_num = 1;
    }

    void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value,
                   size_t num,
                   Param *param = nullptr) override {
        (void)param;
        delete index_;
        index_ = nullptr;

        // SWIX bulk-load expects a stream of (key, timestamp) pairs.
        // IMPORTANT: keep input order (stream order). StreamingBenchmark will provide it time-ordered.
        std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> stream;
        stream.reserve(num);
        for (size_t i = 0; i < num; ++i) stream.push_back(key_value[i]);

        index_ = new swix_t(stream);
    }

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override {
        // SWIX "lookup" returns a count. We expose it via `val` and return (count > 0).
        // Timestamp context comes from Param.ts (set by StreamingBenchmark).
        const PAYLOAD_TYPE ts = static_cast<PAYLOAD_TYPE>(param->ts);
        std::pair<KEY_TYPE, PAYLOAD_TYPE> q{key, ts};
        KEY_TYPE cnt = 0;
        index_->lookup(q, cnt);
        val = static_cast<PAYLOAD_TYPE>(cnt);
        return cnt != 0;
    }

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        (void)param;
        // In streaming benchmark, `value` is the event timestamp for this key.
        std::pair<KEY_TYPE, PAYLOAD_TYPE> ins{key, value};
        index_->insert(ins);
        return true;
    }

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        // SWIX doesn't support in-place update.
        (void)key;
        (void)param;
        return true;
    }

    bool remove(KEY_TYPE key, Param *param = nullptr) override {
        // SWIX uses lazy deletion; GRE test can treat deletes as no-ops.
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

        // SWIX native range_search: (startKey, timestamp, upperBoundKey)
        std::tuple<KEY_TYPE, PAYLOAD_TYPE, KEY_TYPE> q{key_low_bound, ts, upper};

        thread_local std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> buf;
        buf.clear();
        if (buf.capacity() < key_num) buf.reserve(key_num);
        index_->range_search(q, buf);
        return std::min(buf.size(), key_num);
    }

    long long memory_consumption() override {
        return static_cast<long long>(index_->get_total_size_in_bytes());
    }

private:
    swix_t *index_ = nullptr;
};

