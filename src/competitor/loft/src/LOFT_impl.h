#pragma once

#include "LOFT.h"
#include "root_impl.h"
#include "data_node_impl.h"
#include "model_impl.h"
#include "urcu/urcu-qsbr.h"
#include <thread>

namespace loft
{

    template <typename K, typename Val>
    LOFT<K, Val>::LOFT(const std::vector<K> &keys, const std::vector<Val> &vals, size_t worker_num, size_t bg_n) : bg_num(bg_n)
    {
        // to guarantee the corrctness of the input records
        for (size_t key_i = 1; key_i < keys.size(); key_i++)
        {
            assert((keys[key_i] > keys[key_i - 1]));
        }
        work_num = worker_num;
        during_bg = false;
        root = new root_t();
        root->init(keys, vals, work_num);
        start_bg();
    }

    template <typename K, typename Val>
    LOFT<K, Val>::~LOFT()
    {
        terminate_bg();
    }

    template <typename K, typename Val>
    inline bool LOFT<K, Val>::query(const K &key, Val &val, uint8_t & worker_id)
    {
        rcu_read_lock();
        root_t *tmp_root = (root);
        bool thread_counter = during_bg;
        result_t result = tmp_root->query(key, val, worker_id, thread_counter);
        rcu_read_unlock();
        return result == result_t::ok;
    }


    template <typename K, typename Val>
    inline bool LOFT<K, Val>::insert(const K &key, const Val &val, uint8_t & worker_id)
    {
        result_t result;
        rcu_read_lock();
        root_t *tmp_root = (root);
        bool thread_counter = this->during_bg;
        result = tmp_root->insert(key, val, worker_id, thread_counter);
        rcu_read_unlock();
        rcu_quiescent_state();
        
        while (result == result_t::retry)
        {
            rcu_read_lock();
            tmp_root = (root);
            bool thread_counter = this->during_bg;
            result = tmp_root->insert(key, val, worker_id, thread_counter);
            rcu_read_unlock();
            rcu_quiescent_state();
        }
        return result == result_t::ok;
    }

    template <typename K, typename Val>
    inline bool LOFT<K, Val>::update(const K &key, const Val &val, uint8_t & worker_id)
    {
        result_t result;
        rcu_read_lock();
        root_t *tmp_root = rcu_dereference(root);
        bool thread_counter = this->during_bg;
        result = tmp_root->update(key, val, worker_id, thread_counter);
        rcu_read_unlock();
        while (result == result_t::retry)
        {
            rcu_read_lock();
            tmp_root = rcu_dereference(root);
            thread_counter = this->during_bg;
            result = tmp_root->update(key, val, worker_id, thread_counter);
            // std::cout <<"update retry again" << std::endl;
            rcu_read_unlock();
        }
        return result == result_t::ok;
    }

    template <typename K, typename Val>
    inline bool LOFT<K, Val>::remove(const K &key, uint8_t & worker_id)
    {
        rcu_read_lock();
        root_t *tmp_root = rcu_dereference(root);
        bool thread_counter  = this->during_bg;
        result_t result = tmp_root->remove(key, worker_id, thread_counter);
        rcu_read_unlock();
        return result == result_t::ok;
    }

    template <typename K, typename Val>
    inline size_t LOFT<K, Val>::scan(const K &begin, const size_t n,
        std::vector<std::pair<K, Val>> &result, uint8_t & worker_id)
    {
        rcu_read_lock();
        root_t *tmp_root = rcu_dereference(root);
        bool thread_counter  = this->during_bg;
        size_t ans = tmp_root->scan(begin, n, result, thread_counter);
        rcu_read_unlock();
        return ans;
    }

    /*------SMO------*/
    template <typename K, typename Val>
    void LOFT<K, Val>::start_bg()
    {
        bg_running = true;
        bg_master = std::thread(background, this);
    }
    template <typename K, typename Val>
    void LOFT<K, Val>::terminate_bg()
    {
        bg_running = false;
        bg_master.join();
    }
    /*used for check each data node, and trigger model retraining process for better performance */
    template <typename K, typename Val>
    void *LOFT<K, Val>::background(void *this_)
    {

        volatile LOFT &index = *(LOFT *)this_;
        size_t bg_num = index.bg_num;
        if (bg_num == 0)
        {
            return nullptr;
        }        
        std::vector<std::thread> threads;
        std::vector<BGInfo> bg_info(bg_num);

        worksteal *bg_work_queues;
        bg_work_queues = new worksteal[bg_num]();
        // initilize the background thread information
        for (int i = 0; i < bg_num; i++)
        {
            bg_info[i].bg_i = i;
            bg_info[i].bg_n = bg_num;
            bg_info[i].root_ptr = &(index.root);
            bg_info[i].running = true;
            bg_info[i].started = false;
            bg_info[i].finished = false;
            bg_info[i].should_update_array = false;
            bg_info[i].sleep_time = interval_time;
            threads.emplace_back(root_t::do_SMO, &bg_info[i], bg_work_queues);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_time));
        while (index.bg_running)
        {
            root_t *tmp_root = index.root;
            for (size_t j = 0; j < bg_num; j++)
            {

                assert(bg_work_queues[j].empty());
            }
            uint64_t round = tmp_root->data_node_num / bg_num;
            // initialize each work queue
            for (size_t node_num = 0; node_num < round; node_num++)
            {
                for (size_t k = 0; k < bg_num; k++)
                {
                    bg_work_queues[k].push(&(index.root->groups[node_num * bg_num + k].second));
                }
            }
            // insert the remaining nodes pointers
            size_t tmp_loc = round * bg_num;
            while (tmp_loc < tmp_root->data_node_num)
            {
                bg_work_queues[0].push(&(tmp_root->groups[tmp_loc].second));
                tmp_loc++;
            }
            // DEBUG_THIS("---[bg]begin new round of structure update!");
            // start the bg
            for (int i = 0; i < bg_num; i++)
            {
                bg_info[i].started = true;
            }
            index.during_bg = true;
            synchronize_rcu();
            // wait for all bg end
            while (true)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_time/10));
                bool finished = true;
                for (int i = 0; i < bg_num; i++)
                {
                    if (!bg_info[i].finished)
                    {
                        // DEBUG_THIS("---[bg] thread(" << i << ")is not finished");
                        finished = false;
                        break;
                    }
                }

                if (finished)
                {
                    break;
                }
            }
            index.during_bg = false;
            synchronize_rcu();
            // have finsied the bg
            bool update_array = false;
            for (int i = 0; i < bg_num; i++)
            {
                update_array = update_array || bg_info[i].should_update_array;
                bg_info[i].finished = false;
                bg_info[i].should_update_array = false;
            }
            // update the root node, use the
            // use this thread to add and delte data node, concurrency control will be easier
            if (update_array)
            {
                root_t *old_root = index.root;
                root_t *new_root = old_root->create_new_root();
                rcu_xchg_pointer((void **)&(index.root), new_root);
                synchronize_rcu();
                // delete the next pointer
                index.root->clear_next();
                delete old_root;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_time));
        }
        for (int i = 0; i < bg_num; i++)
        {
            bg_info[i].running = false;
        }

        for (int i = 0; i < bg_num; i++)
        {
            threads[i].join();
        }
        return nullptr;
    }

}