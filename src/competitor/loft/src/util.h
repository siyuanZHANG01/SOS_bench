#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>
#include <mutex>
#include <memory>
#include <atomic>
#include "piecewise_linear_model.hpp"
#include "helper.h"
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
#include "jemalloc/jemalloc.h"
#include <immintrin.h> 
#include <random>
#include "zipf.h"

#define EMPTY_KEY 0x0000000000000000
#define READ_HEAVY 0.7
#define WRITE_HEAVY 0.7
#define FILL_LARGE_W 0.9
#define FILL_COLD 0.5
#define FILL_LARGE_R 0.8
#define FILL_LOW 0.4
#define interval_time  100
#define WRITE_HOT_SPOT 0.8
#define ZOOM_COUNTER 256
#define ZOOM_RATIO 0.1
#define FUTURE_FILL 0.9
#define EXPANSION 2
#define ZOOM_FACTOR 8
#define ZOOM_SHARE 32
#define READ_DEGRADATION 0.05
#define DEGRADATION 0.05
#define SHADOW_MASK 0x8000000000000000
#define LOG_MASK 0x4000000000000000
#define LARGE_ZOOMED_LEVEL 2
#define OP_NUM 50
#define CAS(_p, _u, _v)                                              \
    __atomic_compare_exchange_n(_p, _u, _v, false, __ATOMIC_ACQ_REL, \
                                __ATOMIC_ACQUIRE)

#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define NS_PER_S 1000000000.0
#define TIMER_DECLARE(n) struct timespec b##n,e##n
#define TIMER_BEGIN(n) clock_gettime(CLOCK_MONOTONIC, &b##n)
#define TIMER_END_NS(n,t) clock_gettime(CLOCK_MONOTONIC, &e##n); \
    (t)=(e##n.tv_sec-b##n.tv_sec)*NS_PER_S+(e##n.tv_nsec-b##n.tv_nsec)
#define TIMER_END_S(n,t) clock_gettime(CLOCK_MONOTONIC, &e##n); \
    (t)=(e##n.tv_sec-b##n.tv_sec)+(e##n.tv_nsec-b##n.tv_nsec)/NS_PER_S

#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)

#define SELF_INC(val) \
	__asm__ volatile("lock; incl %0;"\
	: "=m" (val)\
	: "m" (val))

struct BGInfo {
  size_t bg_i;  // for calculation responsible range
  size_t bg_n;  // for calculation responsible range
  volatile void *root_ptr;
  volatile bool should_update_array;
  std::atomic<bool> started;
  std::atomic<bool> finished;
  volatile bool running;
  uint64_t sleep_time;
  int thread_id;
};
//segment used for 
template<typename K>
class Segment {
public:
    K key;             ///< The first key that the segment indexes.
    double slope;    ///< The slope of the segment.
    int32_t intercept; ///< The intercept of the segment.

    Segment() = default;

    Segment(K key, double slope, int32_t intercept) : key(key), slope(slope), intercept(intercept) {};

    explicit Segment(size_t n) : key(std::numeric_limits<K>::max()), slope(), intercept(n) {};

    explicit Segment(const typename loft::internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs)
        : key(cs.get_first_x()) {
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(key);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        slope = cs_slope;
        intercept = cs_intercept;
    }

    friend inline bool operator<(const Segment &s, const K &k) { return s.key < k; }
    friend inline bool operator<(const K &k, const Segment &s) { return k < s.key; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.key < t.key; }

    operator K() { return key; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(const K &k) const {
        auto pos = int64_t(slope * (k - key)) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }
};

//functions used for data partition and training the models
template< typename K, typename Segment_vector>
size_t build_level(const std::vector<K> &keys, size_t epsilon, Segment_vector &segments, std::vector<uint64_t> &array_sizes){
    // std::cout <<"keys size is " << keys.size() << std::endl;
    auto first = keys.begin();
    size_t sum = keys.size();
    auto in_fun = [&](auto i) {
        auto x = first[i];
        // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
         // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
        auto flag = i > 0 && i + 1u < sum && x == first[i - 1] && x != first[i + 1] && x + 1 != first[i + 1];
        return std::pair<K, size_t>(x + flag, i);
    };
    auto out_fun = [&](auto cs) { segments.emplace_back(cs); };    

    size_t n_segments = loft::internal::make_segmentation_par(sum, epsilon, in_fun, out_fun, array_sizes);

    if(sum > 1 && segments.back().slope == 0){
        segments.emplace_back(*std::prev(keys.end()) + 1, 0, sum);
        ++n_segments;
    }

    return n_segments;
}

//new functions used for data partition and training the models
template< typename K, typename Segment_vector>
size_t build_level_new(K * keys, size_t epsilon, Segment_vector &segments, std::vector<uint64_t> &array_sizes){
    // std::cout <<"keys size is " << keys.size() << std::endl;
    auto first = keys.begin();
    size_t sum = keys.size();
    //the in fun is used to return the key and the position
    K * keys_buffers = (K *)malloc(50*sizeof(K));
    auto in_fun = [&](auto i) {
        auto x = first[i];
        // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
         // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
        auto flag = i > 0 && i + 1u < sum && x == first[i - 1] && x != first[i + 1] && x + 1 != first[i + 1];
        return std::pair<K, size_t>(x + flag, i);
    };
    auto out_fun = [&](auto cs) { segments.emplace_back(cs); };    

    size_t n_segments = loft::internal::make_segmentation_par(sum, epsilon, in_fun, out_fun, array_sizes);

    if(sum > 1 && segments.back().slope == 0){
        segments.emplace_back(*std::prev(keys.end()) + 1, 0, sum);
        ++n_segments;
    }

    return n_segments;
}
enum class Result;
typedef Result result_t;
enum class Result { ok, failed, retry};

template <class Key>
struct LockKey{
  Key key;
  bool occupied;
  LockKey():occupied(false){}
  LockKey(Key key):key(key),occupied(true){}
};

// ========================= seach-schemes ====================

#define SHUF(i0, i1, i2, i3) (i0 + i1*4 + i2*16 + i3*64)
#define FORCEINLINE __attribute__((always_inline)) inline

// power of 2 at most x, undefined for x == 0
FORCEINLINE uint32_t bsr(uint32_t x) {
  return 31 - __builtin_clz(x);
}

template<typename GROUP_TYPE, typename KEY_TYPE>
static int binary_search_branchless(const std::vector<GROUP_TYPE> & arr, int n, KEY_TYPE key) {
//static int binary_search_branchless(const int *arr, int n, int key) {
  intptr_t pos = -1;
  intptr_t logstep = bsr(n - 1);
  intptr_t step = intptr_t(1) << logstep;

  pos = (arr[pos + n - step].first < key ? pos + n - step : pos);
  step >>= 1;

  while (step > 0) {
    pos = (arr[pos + step].first < key ? pos + step : pos);
    step >>= 1;
  }
  pos += 1;

  return (int) (arr[pos].first >= key ? pos : n);
}


//segment used for 
template<typename K, typename val>
class tmp_log {
  typedef std::pair<K, val *> record;
public:
    record * log_slot = nullptr;
    record * tail = nullptr;
    size_t loc = 0;
    size_t init_size = 0;
    record * end = nullptr;
    tmp_log * next = nullptr;
    tmp_log() = default;
    tmp_log(size_t n){
      init_size = n;
      size_t size = (16 << init_size);
      log_slot = new record[size];      
      tail = log_slot;
      end = log_slot + size;
    }

    void push_back(K key, val* value){
      while(1){
        if(next == nullptr){
          record * tmp_ptr = this->tail;
          if(tmp_ptr == end){
            tmp_log * next_log = new tmp_log(init_size + 1);
            tmp_log * null_log = nullptr;
            if(!CAS(&(this->next), &null_log, next_log)){
                return null_log->push_back(key, value);
              }
              return this->next->push_back(key, value);
          }
          record * loc_ptr = this->tail;
          record * next_ptr = tmp_ptr+1;
          if(CAS(&this->tail, &tmp_ptr, next_ptr)){
            *loc_ptr = std::make_pair(key, value);
            break;
          }
        }else{
          next->push_back(key, value);
          return ;
        }
        
      }
    }

    bool pop_back(K & key, val & value){
      if(&log_slot[loc] == end && this->next != nullptr){
        return this->next->pop_back(key, value);
      }
      if(&log_slot[loc] == tail){
         return false;
      }
      key = log_slot[loc].first;
      value = *(log_slot[loc].second);
      loc++;
      return true;
    }

    void free(){
      tmp_log * next_log = next;
      if(next_log != nullptr){
        next_log->free();
      }
      delete [] log_slot;
      return ;
      }
};



int find_first_num_avx2(size_t key, uint64_t *array, int size) {
    // int i = 0;
    // __m256i A_vec = _mm256_set1_epi64x(key);
    // // Process 4 uint64_t elements (256 bits) at a time
    // for (; i + 4 <= size; i += 4) {
    //     // Load 4 uint64_t values into an AVX2 register
    //     __m256i data = _mm256_loadu_si256((__m256i*)&array[i]);

    //     __m256i cmpA = _mm256_cmpeq_epi64(data, A_vec);
    //     // Convert the comparison result into a bitmask
    //     int mask = _mm256_movemask_epi8(cmpA);

    //     // If the mask is non-zero, find the first zero element
    //     if (mask != 0) {
    //         // Find the index of the first zero within the chunk
    //         int zero_index = __builtin_ctz(mask) / 8;  // Find the first 1-bit (8 bytes per element)
    //         return i + zero_index;
    //     }
    // }

    // // Check the remainder of the array (if any)
    // for (; i < size; i++) {
    //     if (array[i] == 0) {
    //         return i;
    //     }
    // }
    std::cout << "do not support that" << std::endl;
    // Return -1 if no zero is found
    return -1;
}

template<class T>
T *get_search_keys(T array[], int num_keys, int num_searches, size_t *seed = nullptr) {
    auto *keys = new T[num_searches];
    {
        std::mt19937_64 gen(std::random_device{}());
        if (seed) {
            gen.seed(*seed + 100);
        }
        std::uniform_int_distribution<int> dis(0, num_keys - 1);
        for (int i = 0; i < num_searches; i++) {
            int pos = dis(gen);
            keys[i] = array[pos];
        }
    }

    return keys;
}

template<class T>
T *get_search_keys_zipf(T array[], int num_keys, int num_searches, size_t *seed = nullptr) {
    auto *keys = new T[num_searches];
    ScrambledZipfianGenerator zipf_gen(num_keys, seed);
    for (int i = 0; i < num_searches; i++) {
        int pos = zipf_gen.nextValue();
        assert(pos < num_keys);
        keys[i] = array[pos];
    }
    return keys;
}

