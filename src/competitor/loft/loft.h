#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>

#include "../indexInterface.h"

// LOFT uses userspace RCU QSBR. In some environments (e.g., Windows), the headers/libs may be absent.
// Prefer the build-system decision (GRE_HAVE_URCU_QSBR). If it isn't defined, fall back to header probing.
#if defined(GRE_HAVE_URCU_QSBR)
#if GRE_HAVE_URCU_QSBR
#define GRE_LOFT_ENABLED 1
#else
#define GRE_LOFT_ENABLED 0
#endif
#elif __has_include(<urcu/urcu-qsbr.h>) || __has_include("urcu/urcu-qsbr.h")
#define GRE_LOFT_ENABLED 1
#else
#define GRE_LOFT_ENABLED 0
#endif

#if GRE_LOFT_ENABLED

#include "./src/LOFT.h"
#include "./src/LOFT_impl.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class loftInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    loftInterface() = default;

    ~loftInterface() override {
        delete index_;
        index_ = nullptr;
    }

    void init(Param *param = nullptr) override {
        // LOFT requires a per-thread urcu registration, but GRE's benchmark does not call init()
        // on every worker thread. We therefore do lazy per-thread registration in each operation.
        if (param) {
            worker_num_ = param->worker_num;
        }
        // Similar to xindex: reserve a few background threads for retraining.
        bg_n_ = std::max<size_t>(1, worker_num_ / 12 + 1);

        // LOFT's worker_id is uint8_t; LOFT microbench constructs work_num = fg_n + 1.
        work_num_ = worker_num_ + 1;
        if (work_num_ > 255) {
            std::cerr << "[loft] error: work_num (" << work_num_
                      << ") exceeds uint8_t worker_id range; reduce thread_num.\n";
            std::exit(1);
        }
    }

    void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *key_value, size_t num, Param *param = nullptr) override {
        (void)param;
        if (index_ != nullptr) {
            delete index_;
            index_ = nullptr;
        }

        std::vector<KEY_TYPE> keys;
        std::vector<PAYLOAD_TYPE> vals;
        keys.reserve(num);
        vals.reserve(num);

        // Keep this aligned with other GRE adapters: just adapt the bulk-load array into LOFT's
        // required input vectors. (GRE benchmark already sorts init keys before bulk_load.)
        for (size_t i = 0; i < num; i++) {
            keys.push_back(key_value[i].first);
            vals.push_back(key_value[i].second);
        }

        // Ensure init() was called at least once to size work/bg params.
        if (worker_num_ == 0) worker_num_ = 1;
        if (bg_n_ == 0) bg_n_ = 1;
        if (work_num_ == 0) work_num_ = worker_num_ + 1;

        index_ = new loft::LOFT<KEY_TYPE, PAYLOAD_TYPE>(keys, vals, work_num_, bg_n_);
    }

    bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) override {
        ensure_rcu_registered();
        uint8_t &wid = tls_worker_id(param);
        return index_->query(key, val, wid);
    }

    bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        ensure_rcu_registered();
        uint8_t &wid = tls_worker_id(param);
        return index_->insert(key, value, wid);
    }

    bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) override {
        ensure_rcu_registered();
        uint8_t &wid = tls_worker_id(param);
        return index_->update(key, value, wid);
    }

    bool remove(KEY_TYPE key, Param *param = nullptr) override {
        ensure_rcu_registered();
        uint8_t &wid = tls_worker_id(param);
        return index_->remove(key, wid);
    }

    size_t scan(KEY_TYPE key_low_bound,
                size_t key_num,
                std::pair<KEY_TYPE, PAYLOAD_TYPE> *result,
                Param *param = nullptr) override {
        (void)result;
        ensure_rcu_registered();
        uint8_t &wid = tls_worker_id(param);

        thread_local std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> buf;
        buf.clear();
        if (buf.capacity() < key_num) buf.reserve(key_num);

        // Align with other GRE adapters (e.g., finedex): the benchmark only uses the returned
        // length; it does not consume the scan result contents.
        return index_->scan(key_low_bound, key_num, buf, wid);
    }

    long long memory_consumption() override {
        return 0;
    }

private:
    loft::LOFT<KEY_TYPE, PAYLOAD_TYPE> *index_ = nullptr;
    size_t worker_num_ = 1;
    size_t bg_n_ = 1;
    size_t work_num_ = 2;

    static void ensure_rcu_registered() {
        // LOFT requires urcu registration per calling thread.
        thread_local bool registered = false;
        if (!registered) {
            rcu_register_thread();
            registered = true;
        }
    }

    uint8_t &tls_worker_id(Param *param) {
        thread_local bool inited = false;
        thread_local uint8_t wid = 0;
        if (!inited) {
            uint32_t tid = param ? param->thread_id : 0;
            wid = static_cast<uint8_t>(tid % static_cast<uint32_t>(work_num_));
            inited = true;
        }
        return wid;
    }
};

#else  // GRE_LOFT_ENABLED == 0

template<class KEY_TYPE, class PAYLOAD_TYPE>
class loftInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
public:
    void bulk_load(std::pair<KEY_TYPE, PAYLOAD_TYPE> *, size_t, Param * = nullptr) override {
        std::cerr << "[loft] error: LOFT is not enabled (missing urcu-qsbr).\n";
        std::exit(1);
    }
    bool get(KEY_TYPE, PAYLOAD_TYPE &, Param * = nullptr) override { return false; }
    bool put(KEY_TYPE, PAYLOAD_TYPE, Param * = nullptr) override { return false; }
    bool update(KEY_TYPE, PAYLOAD_TYPE, Param * = nullptr) override { return false; }
    bool remove(KEY_TYPE, Param * = nullptr) override { return false; }
    size_t scan(KEY_TYPE, size_t, std::pair<KEY_TYPE, PAYLOAD_TYPE> *, Param * = nullptr) override { return 0; }
    void init(Param * = nullptr) override {}
    long long memory_consumption() override { return 0; }
};

#endif