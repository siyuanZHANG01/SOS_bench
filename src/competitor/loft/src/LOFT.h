#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <thread>
#include <iomanip>
#include <optional>

#include "util.h"
#include "helper.h"
#include "root.h"
#include "model.h"
#include "data_node.h"
#include "work_stealing.h"


namespace loft{
template<typename K, typename Val >
class LOFT {
    typedef Root<K, Val> root_t;
    typedef Dnode<K,Val> Dnode_;
    typedef WorkStealingQueue<Dnode_ * volatile*> worksteal;

public:
    
    LOFT(const std::vector<K> &keys, const std::vector<Val> &vals, size_t work_num, size_t bg_n);
    ~LOFT();

    //different kinds of operations
    inline bool query(const K &key, Val &val, uint8_t & worker_id);
    inline bool insert(const K &key, const Val &val, uint8_t & worker_id);
    inline bool update(const K &key, const Val &val, uint8_t & worker_id);
    inline bool remove(const K &key, uint8_t & worker_id);
    inline size_t scan(const K &begin, const size_t n,
        std::vector<std::pair<K, Val>> &result, uint8_t & worker_id);

private:
    void start_bg(); 
    void terminate_bg(); 
    static void *background(void *this_);
    volatile bool bg_running = true;
    volatile bool during_bg = false;
    std::thread bg_master;
    size_t bg_num = 1;
    size_t work_num;
    root_t *volatile root = nullptr;
    
};
}