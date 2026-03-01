#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "../indexInterface.h"

// CARMI provides an STL-map-like container: CARMIMap<Key, Value>.
// This adapter intentionally forces single-thread execution in GRE benchmark
// (CARMI codebase shows no explicit concurrency control primitives).
#include "./src/src/include/carmi_map.h"
// CARMIMap::CalculateSpace() calls CARMI::CalculateSpace(), whose template
// definition lives in a separate header in the upstream codebase. Include it
// here to ensure the symbol is available when we link microbench.
#include "./src/src/include/func/calculate_space.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class carmiInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    using carmi_t = CARMIMap<KEY_TYPE, PAYLOAD_TYPE>;

    carmiInterface() = default;

    ~carmiInterface() override {
        delete index_;
        index_ = nullptr;
    }

    void init(Param *param = nullptr) override {
        // CARMIMap is not designed for concurrent mutation without external synchronization.
        // Align with GRE's mechanism: shrink worker_num so benchmark runs single-thread.
        if (param) param->worker_num = 1;
    }

    void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) override {
        (void)param;
        delete index_;
        index_ = nullptr;

        // CARMIMap range constructor: it will preprocess/sort internally if needed.
        index_ = new carmi_t(key_value, key_value + num);
    }

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override {
        (void)param;
        auto it = index_->find(key);
        if (it == index_->end()) return false;
        val = it.data();
        return true;
    }

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        (void)param;
        return index_->insert({key, value}).second;
    }

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        (void)param;
        auto it = index_->find(key);
        if (it == index_->end()) return false;
        it.data() = value;
        return true;
    }

    bool remove(KEY_TYPE key, Param *param = nullptr) override {
        (void)param;
        return index_->erase(key) > 0;
    }

    size_t scan(KEY_TYPE key_low_bound,
                size_t key_num,
                std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr) override {
        (void)param;
        if (!result || key_num == 0) return 0;

        size_t got = 0;
        for (auto it = index_->lower_bound(key_low_bound);
             it != index_->end() && got < key_num;
             ++it) {
            // Fill GRE's output buffer directly (one construction, no temp vector).
            result[got] = {it.key(), it.data()};
            ++got;
        }
        return got;
    }

    long long memory_consumption() override {
        if (!index_) return 0;
        // CARMI exposes a built-in space estimator for the index structure.
        // It is intended to report the in-memory footprint in bytes.
        return index_->CalculateSpace();
    }

private:
    carmi_t *index_ = nullptr;
};

