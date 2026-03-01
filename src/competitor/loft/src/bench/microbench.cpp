#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "LOFT_impl.h"
#include "util.h"
#include "LOFT.h"
#include <sstream>
#include <string>
#include <fstream>
#include <getopt.h>
#include <malloc.h>
#include <random>
#include <time.h>

#define CACHELINE_SIZE (1 << 6)
#define KEY_TYPE uint64_t
#define VAL_TYPE uint64_t
inline void load_data();
inline void parse_args(int, char **);

void uniform_data();
void normal_data();
void lognormal_data();

struct alignas(CACHELINE_SIZE) FGParam;
std::atomic<size_t> ready_threads(0);
typedef struct operation_item {
    KEY_TYPE key;
    int32_t range;
    uint8_t op;
    } operation_item;
std::vector <operation_item> operations;

typedef FGParam fg_param_t;
volatile bool running = false;
typedef loft::LOFT<KEY_TYPE,VAL_TYPE> loft_index;
std::vector<KEY_TYPE> exist_keys;
std::vector<KEY_TYPE> non_exist_keys;
// parameters
size_t fg_n = 1;
int init_data_num = 1000000;
int data_num = 20000000;
uint64_t oper_num = 1000000;
size_t bg_n = 1;
size_t benchmark = 1;
size_t run_time = 5;
double read_ratio = 1;
double insert_ratio = 0;
double delete_ratio = 0;
double update_ratio = 0;
double scan_ratio = 0;
double hot_spot = 0.1;
uint8_t zipf = 1;
uint64_t min_key_in_dataset;
struct alignas(CACHELINE_SIZE) FGParam {
  loft_index *table;
  uint64_t throughput;
  uint32_t thread_id;
  char padding[CACHELINE_SIZE - sizeof(loft_index*) - sizeof(uint64_t) - sizeof(uint32_t)];
};
std::mt19937 gen;


void generate_operations() {
    size_t query_i = 0, insert_i = 0, delete_i = 0, update_i = 0, scan_i = 0;
    size_t item_num = 0;
    uint64_t key;
    uint64_t dummy_key = 1234;
    operations.reserve(oper_num);
    uint64_t oper_num_tmp = 0;
    if(insert_ratio == 1){
        for(int j = 0; j < data_num - init_data_num; j ++){
                operation_item op_tmp;
                op_tmp.op = 1;
                op_tmp.key = non_exist_keys[j];
                operations.push_back(op_tmp);
                oper_num_tmp++;
                insert_i++;
            }
    }else{
        KEY_TYPE *sample_ptr = nullptr;
        size_t random_seed = 1866;
        if (zipf == 0) {
            sample_ptr = get_search_keys(&exist_keys[0], init_data_num, oper_num, &random_seed);
        } else {
            sample_ptr = get_search_keys_zipf(&exist_keys[0], init_data_num, oper_num, &random_seed);
        }
        COUT_THIS("generate operations.");
        std::uniform_real_distribution<> ratio_dis(0, 1);
        operation_item op_tmp;
        size_t sample_counter = 0;
        size_t temp_counter = 0;
        size_t insert_counter = 0;
        for(int j = 0; j < oper_num; j++){
            auto prob = ratio_dis(gen);
            if (prob < read_ratio) {
                op_tmp.op = 0;
                op_tmp.key = sample_ptr[sample_counter++];
                operations.push_back(op_tmp);
                query_i++;
            }else{
                if (insert_counter >= non_exist_keys.size()) {
                    oper_num = j;
                    break;
                }
                 op_tmp.op = 1;
                op_tmp.key = non_exist_keys[insert_counter];
                operations.push_back(op_tmp);
                insert_counter++;
                insert_i++;
            }
            oper_num_tmp++;
        }
            
    }
      oper_num = oper_num_tmp;
      COUT_VAR(operations.size());
      COUT_VAR(query_i);
      COUT_VAR(insert_i);

    }

template <class T>
bool load_binary_data(T data[], int length, const std::string& file_path) {
  std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is.is_open()) {
    std::cout <<" can not open" << std::endl;
    return false;
  }
  double * tmp_data = new double[length];
  is.read(reinterpret_cast<char*>(tmp_data), std::streamsize(length * sizeof(T)));
  for(int i = 0; i < length; i++){
    data[i] = (uint64_t)((tmp_data[i] + 200) * 1000000000000UL);
  }
  is.close();
  return true;
}


void *run_fg(void *param) {
  fg_param_t &thread_param = *(fg_param_t *)param;
  uint32_t thread_id = thread_param.thread_id;
  loft_index *table = thread_param.table;
  uint8_t worker_i = (uint32_t)thread_param.thread_id;
  size_t key_n_per_thread = oper_num / fg_n;
  size_t key_start = thread_id * key_n_per_thread;
  size_t key_end = (thread_id + 1) * key_n_per_thread;
  VAL_TYPE dummy_value = 1234;
  int i = key_start;
  size_t operations_done = 0;
  
  // Use timespec for more efficient timing
  struct timespec start_time, end_time;
  
  rcu_register_thread();
  ready_threads++;
  while (!running)
    ;

  // Get start time once
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  
  while(i < key_end) {
    operation_item opi = operations[i];
    __builtin_prefetch(&operations[i + 8]);  // Prefetch future operations
    if(opi.op == 0) {     // read
      if(opi.key < min_key_in_dataset) {
        i++;
        continue;
      }
      if(table->query(opi.key, dummy_value, worker_i)) {
        operations_done++;
      }
    } else if (opi.op == 1) {
      if(opi.key < min_key_in_dataset) {
        i++;
        continue;
      }
      if(table->insert(opi.key, dummy_value, worker_i)) {
        operations_done++;
      }
    }
    i++;
  }

  // Get end time once
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  
  rcu_unregister_thread();

  // Calculate elapsed time in microseconds
  uint64_t elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000;
                       
  // Calculate throughput
  double thread_throughput = (double)(key_end - key_start) / (double)elapsed_us * 1000000.0;
  
  thread_param.throughput = thread_throughput;
  // std::cout << "Thread " << thread_id << " throughput: " << thread_throughput << " ops/sec" << std::endl;
  
  pthread_exit(nullptr);
}

int main(int argc, char ** argv){
    // Generate some random data
    // parse_args(argc, argv);
    load_data();
    generate_operations();
    std::vector<VAL_TYPE> vals(init_data_num);
    std::sort(exist_keys.begin(), exist_keys.end());
    //to verify the rightness
    for(int i = 0; i < vals.size(); i++){
      vals[i]=(VAL_TYPE)i;
    }

    sleep(1);
    double time_s = 0.0;
    //construct the index
    loft_index *  index_ = new loft_index(exist_keys,vals,fg_n + 1,bg_n);
    pthread_t threads[fg_n];
    fg_param_t fg_params[fg_n];
    
    running = false;
    for (size_t worker_i = 0; worker_i < fg_n; worker_i++) {
      fg_params[worker_i].table = index_;
      fg_params[worker_i].thread_id = worker_i;
      fg_params[worker_i].throughput = 0;
      int ret = pthread_create(&threads[worker_i], nullptr, run_fg,
                             (void *)&fg_params[worker_i]);
      
      if (ret) {
        COUT_N_EXIT("Error:" << ret);
      }
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(worker_i + bg_n + 1, &cpuset);
      int rc = pthread_setaffinity_np(threads[worker_i],sizeof(cpu_set_t), &cpuset);
      if(rc!=0){
        std::cout<<"SET  CPU ERROR!\n";
      }
    }
    std::vector<size_t> tput_history(fg_n, 0);

    COUT_THIS("[micro] prepare data ...");
    while (ready_threads < fg_n) sleep(1);
    running = true;

    void *status;
    for (size_t i = 0; i < fg_n; i++) {
      int rc = pthread_join(threads[i], &status);
      if (rc) {
        COUT_N_EXIT("Error:unable to join," << rc);
      }
    }

    size_t throughput = 0;
    for (auto &p : fg_params) {
      throughput += p.throughput;
    }
    COUT_THIS("[micro] Throughput(op/s): " << throughput);
    index_->~LOFT();
    std::cout <<"the master thread has already end" << std::endl;
    return 0;
}



inline void parse_args(int argc, char **argv) {
  struct option long_options[] = {
      {"read", required_argument, 0, 'a'},
      {"insert", required_argument, 0, 'b'},
      {"remove", required_argument, 0, 'c'},
      {"update", required_argument, 0, 'd'},
      {"init_num", required_argument, 0, 'e'},
      {"zipf", required_argument, 0, 'f'},
      {"fg_n", required_argument, 0, 'g'},
      {"benchmark", required_argument, 0, 'h'},
      {"bg_n", required_argument, 0, 'i'},
      {"data_num", required_argument, 0, 'j'},
      {"oper_num", required_argument, 0, 'k'},
      {0, 0, 0, 0}};
  std::string ops = "a:b:c:d:e:f:g:h:i:j:k:";
  int option_index = 0;

  while (1) {
    int c = getopt_long(argc, argv, ops.c_str(), long_options, &option_index);
    if (c == -1) break;
    switch (c) {
      case 0:
        if (long_options[option_index].flag != 0) break;
        abort();
        break;
      case 'a':
        read_ratio = strtod(optarg, NULL);
        INVARIANT(read_ratio >= 0 && read_ratio <= 1);
        break;
      case 'b':
        insert_ratio = strtod(optarg, NULL);
        INVARIANT(insert_ratio >= 0 && insert_ratio <= 1);
        break;
      case 'c':
        delete_ratio = strtod(optarg, NULL);
        INVARIANT(delete_ratio >= 0 && delete_ratio <= 1);
        break;
      case 'd':
        update_ratio = strtod(optarg, NULL);
        INVARIANT(update_ratio >= 0 && update_ratio <= 1);
        break;
      case 'e':
        init_data_num = strtoul(optarg, NULL, 10);
        INVARIANT(init_data_num > 0);
        break;
      case 'f':
        zipf = strtoul(optarg, NULL, 10);
        break;
      case 'g':
        fg_n = strtoul(optarg, NULL, 10);
        INVARIANT(fg_n > 0);
        break;
      case 'h':
        benchmark = strtoul(optarg, NULL, 10);
        INVARIANT(benchmark >= 0 && benchmark<14);
        break;
      case 'i':
        bg_n = strtoul(optarg, NULL, 10);
        INVARIANT(bg_n >= 0);
        break;
      case 'j':
        data_num = strtoul(optarg, NULL, 10);
        INVARIANT(data_num > 0);
        break;
      case 'k':
        oper_num = strtoul(optarg, NULL, 10);
        INVARIANT(oper_num > 0);
        break;
      default:
        abort();
    }
  }

  COUT_THIS("[micro] Read:Insert:Update:Delete:Scan = "
            << read_ratio << ":" << insert_ratio << ":" << update_ratio << ":"
            << delete_ratio );
  double ratio_sum =
      read_ratio + insert_ratio + delete_ratio + update_ratio;
  INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);  // avoid precision lost
  COUT_VAR(fg_n);
  COUT_VAR(benchmark);
}

void load_data(){
    switch (benchmark) {
    case 0:
      normal_data();
        break;
    case 1:
        lognormal_data();
        break;
    case 2:
        uniform_data();
        break;
	default:
		abort();
    }

    // initilize XIndex (sort keys first)
    COUT_THIS("[processing data]");
    std::sort(exist_keys.begin(), exist_keys.end());
    auto last = std::unique(exist_keys.begin(), exist_keys.end());
    exist_keys.erase(last, exist_keys.end());
    //preserve the unqique keys
    for(size_t i=1; i<exist_keys.size(); i++){
        assert(exist_keys[i]>exist_keys[i-1]);
    }
    min_key_in_dataset = exist_keys[0];
    COUT_VAR(exist_keys.size());
    COUT_VAR(non_exist_keys.size());
}


void uniform_data(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::srand(time(NULL));
    std::uniform_int_distribution<KEY_TYPE> rand_uniform(0, data_num);
    exist_keys.reserve(init_data_num);
    for (size_t i = 0; i < init_data_num; ++i) {
        exist_keys.push_back(rand_uniform(gen));
    }
    if (insert_ratio > 0) {
        non_exist_keys.reserve(data_num - init_data_num);
        for (size_t i = init_data_num; i < data_num; ++i) {
            non_exist_keys.push_back(rand_uniform(gen));
        }
    }
}

void normal_data(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> rand_normal(4, 2);

    exist_keys.reserve(init_data_num);
    for (size_t i = 0; i < init_data_num; ++i) {
        KEY_TYPE a = rand_normal(gen)*1000000000000;
        if(a<0) {
            i--;
            continue;
        }
        exist_keys.push_back(a);
    }
    if (insert_ratio > 0) {
        non_exist_keys.reserve(data_num - init_data_num);
        for (size_t i = 0; i < data_num - init_data_num; ++i) {
            KEY_TYPE a = rand_normal(gen)*1000000000000;
            if(a<0) {
                i--;
                continue;
            }
            non_exist_keys.push_back(a);
        }
    }
}
void lognormal_data(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<double> rand_lognormal(0, 2);

    exist_keys.reserve(init_data_num);
    for (size_t i = 0; i < init_data_num; ++i) {
        KEY_TYPE a = rand_lognormal(gen)*1000000000000;
        assert(a>0);
        exist_keys.push_back(a);
    }
    if (insert_ratio > 0) {
        non_exist_keys.reserve(data_num - init_data_num);
        for (size_t i = 0; i < data_num - init_data_num; ++i) {
            KEY_TYPE a = rand_lognormal(gen)*1000000000000;
            assert(a>0);
            non_exist_keys.push_back(a);
        }
    }
}


