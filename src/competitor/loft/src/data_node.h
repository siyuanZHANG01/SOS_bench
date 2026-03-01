#pragma once
#include "model.h"
#include <ctime>
#include "util.h"
#include <atomic>

#include <cmath>
namespace loft{

template<typename K, typename Val >
class alignas(CACHELINE_SIZE) Dnode {
    template<typename Key, typename Vals> friend class LOFT;
    template<typename Key, typename Vals> friend class Root;
    typedef LinearRegressionModel<K> linear_model_t;
    typedef Segment<K> segment;
    typedef LockKey<K> lock_key;
    typedef std::pair<K, Val> record;
    typedef Val* val_ptr_t;

public: 
    Dnode();
    ~Dnode();
    void init(const std::vector<K> & key_begin,const std::vector<Val> & val_begin, double expansion, size_t epsilon, segment  seg, uint64_t array_size, size_t begin_addr, size_t read_epsilon, size_t worker_num);
    inline result_t query(const K &key, Val &val, uint8_t & worker_id, bool countered);
    inline result_t query_shadownode(const K &key, Val &val, uint8_t & worker_id, bool counter);
    inline result_t query(const K &key, Val &val, val_ptr_t & val_ptr, uint8_t & worker_id);
    inline result_t insert(const K &key, const Val &val, uint8_t & worker_id, bool counter);

    inline result_t update(const K &key, const Val &val, uint8_t & worker_id, bool counter);
    inline result_t remove(const K &key, uint8_t & worker_id, bool counter);
    inline size_t scan(const K &begin, const size_t n,
        std::vector<std::pair<K, Val>> &result, bool counter);

    void print_args();
    K get_pivot();
    //for model retraining
    Dnode *split_data_node();
    Dnode *merge_data_node(Dnode * next_node);
    void sync_reocrd();
    Dnode * compact_phase();//first merge and retraining 
    Dnode * new_compact_phase();//first merge and retraining
    inline bool sync_log(const K &key, const Val &val);
    void free_data();
    void free_log();


    uint64_t init_amount();
    uint64_t array_size();
    
private:
    K min_key;
    size_t array_size_;
    //history information
    double expansion_ = 1.5;
    size_t epsilon_ = 32;
    size_t read_eplision_ = 32;

    bool sync_phase = false;
    bool zoomed = false;
    bool retraining_phase = false;
    

    K ** zoomed_keys = nullptr;
    Val ** zoomed_vals = nullptr;
    uint64_t zoomed_level = 0;   
    
    K * keys = nullptr;
    Val * vals = nullptr;
    //model part
    linear_model_t * model = nullptr;

    //add the counter and timer
    uint64_t * counter_r;
    uint64_t * counter_w;
    std::atomic<uint64_t> before_retrain_counter = 0;
    std::atomic<uint64_t> during_retrain_counter = 0;
    uint64_t init_size = 0;
    uint64_t init_buffer_size = 0;
    // to record the create time 
    
    size_t worker_num = 0;
    uint64_t history_amount = 0; 
    uint64_t history_w_1 = 0;
    uint64_t history_w_2 = 0;
    uint64_t history_r_1 = 0;
    uint64_t history_r_2 = 0;
    uint64_t history_r_3 = 0;

    Dnode * next = nullptr; // for model retraining
    Dnode * last = nullptr;
    Dnode * training_node = nullptr; // for model retraining
    tmp_log<K, Val> * write_log = nullptr; //used for log the new insert

    //for create new node with pointer
    bool write_intensive = false;
    bool read_intensive = false;
    bool cold = false;
    bool skew = false;
    bool new_retrained = false;
    void expansion(const std::vector<K> & key_begin,const std::vector<Val> & val_begin, double expansion,  size_t epsilon, segment  seg, uint64_t array_size,  size_t read_epsilon);
    void expansion_new(Dnode * retrained_node, double expansion,  size_t epsilon, segment  seg,  size_t read_epsilon);
    bool insert_record(const K &key, const Val &val, bool & bucket);
    bool insert_to_bucket(const K &key, const Val &val, size_t loc, size_t predict_po);
    void insertion_search_sort(std::vector<K> & new_keys,std::vector<Val> & new_vals, size_t & n);
    void insertion_search(std::vector<K> & new_keys,std::vector<Val> & new_vals, size_t & n);
    inline result_t insert_to_expand_bucket(const K &key, const Val &val, size_t loc, size_t predict_po, uint8_t & worker_id );
    bool cmp(std::pair<K, Val> R1, std::pair<K, Val> R2);

};   
}