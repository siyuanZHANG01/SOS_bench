#pragma once
#include "data_node.h"
#include "util.h"
#include <memory>
#include <algorithm>
#include <unistd.h>
#include "urcu/urcu-qsbr.h"
#include "work_stealing.h"

#define rcu_read_lock           urcu_qsbr_read_lock
#define _rcu_read_lock          _urcu_qsbr_read_lock
#define rcu_read_unlock         urcu_qsbr_read_unlock
#define _rcu_read_unlock        _urcu_qsbr_read_unlock
#define rcu_read_ongoing        urcu_qsbr_read_ongoing
#define _rcu_read_ongoing       _urcu_qsbr_read_ongoing
#define rcu_quiescent_state     urcu_qsbr_quiescent_state
#define _rcu_quiescent_state        _urcu_qsbr_quiescent_state
#define rcu_thread_offline      urcu_qsbr_thread_offline
#define rcu_thread_online       urcu_qsbr_thread_online
#define rcu_register_thread     urcu_qsbr_register_thread
#define rcu_unregister_thread       urcu_qsbr_unregister_thread
#define rcu_exit            urcu_qsbr_exit
#define synchronize_rcu         urcu_qsbr_synchronize_rcu
#define rcu_reader          urcu_qsbr_reader
#define rcu_gp              urcu_qsbr_gp

#define root_error_bound 8

namespace loft{
template<typename K, typename Val >
class Root{
  typedef Dnode<K,Val> Dnode_;
  typedef LinearRegressionModel<K> linear_model_t;
  typedef Segment<K> segment;
  typedef WorkStealingQueue<Dnode_ * volatile*> worksteal;
  typedef Val* val_ptr_t;
  template<typename Key, typename Vals> friend class LOFT;
public:
  Root();
  ~Root();
  void init(const std::vector<K> &keys, const std::vector<Val> &vals, const size_t worker_num);
  //Do operations
  inline result_t query(const K &key, Val &val, uint8_t & worker_id, bool countered);
  inline result_t insert(const K &key, const Val &val, uint8_t & worker_id, bool countered);
  inline result_t update(const K &key, const Val &val, uint8_t & worker_id, bool countered);
  inline result_t remove(const K &key, uint8_t & worker_id, bool countered);
  inline size_t scan(const K &begin, const size_t n,
        std::vector<std::pair<K, Val>> &result, bool countered);

  //Do SMO
  Root *create_new_root();
  void trim_root();
  static void * do_SMO(void *args, worksteal * work_queue);
  void free_root();
  void clear_next();
private:
  Dnode_ *locate_group_3(const K &key, size_t & begin_node, bool & linked_node);
  
  size_t predict(const K &key);
  void train_rmi_xindex();
  void train_rmi_xindex_2(size_t rmi_2nd_stage_model_n, bool skew);
  size_t pick_next_stage_model(size_t group_i_pred);

 // for SMO   
  void estimate_overhead_dynamically(bool &split, bool & merge, bool &expand, Dnode_ * e_Dnode);


  bool execute_smo(size_t &split, size_t & merge, size_t &expand, Dnode_ * volatile* Data_node,Root &root);
  
  size_t init_epsilon =  32;
  double init_expansion = 1.5;
  size_t init_read_epsilon = 32;
  size_t data_node_num = 0; 
  size_t worker_num = 0;
  //used for searching data node
  bool skewd_workloads = false;
  linear_model_t rmi_1st_stage;
  linear_model_t rmi_1st_stage_backup;
  linear_model_t * rmi_2nd_stage = nullptr; 
  size_t rmi_2nd_stage_model_n = 0;

  std::vector<segment> root_rmi;
  //data node
  std::unique_ptr<std::pair<K, Dnode_ *volatile>[]> groups;
  bool cmp(const std::pair<K, Val> a, const std::pair<K, Val> b);


};
}
