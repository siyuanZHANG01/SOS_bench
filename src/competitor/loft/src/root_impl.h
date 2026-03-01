#pragma once
#include "root.h"


namespace loft
{
    template <typename K, typename Val>
    Root<K, Val>::~Root() {}
    template <typename K, typename Val>
    Root<K, Val>::Root() {}
    template <typename K, typename Val>
    bool cmp(const std::pair<K, Val> a, const std::pair<K, Val> b){
        return a.first < b.first;
    }
    // initial the whole index, in another word, bulk load a batch of records
    template <typename K, typename Val>
    void Root<K, Val>::init(const std::vector<K> &keys, const std::vector<Val> &vals, const size_t worker_num)
    {
        size_t key_num = keys.size();
        this->worker_num = worker_num;
        DEBUG_THIS("[ROOT]: the key num is " << key_num);
        DEBUG_THIS("[ROOT]: the worker num is "<< worker_num - 1);
        // get the segment num and the exact partition
        std::vector<uint64_t> array_sizes;
        array_sizes.reserve(key_num / (init_epsilon * init_epsilon));
        root_rmi.reserve(key_num / (init_epsilon * init_epsilon));
        data_node_num = build_level(keys, init_epsilon, root_rmi, array_sizes);

        DEBUG_THIS("[ROOT]: Begin root initialization!");
        // replace all data, initialize each group
        size_t begin_addr = 0;
        groups = std::make_unique<std::pair<K, Dnode_ *volatile>[]>(data_node_num);
        for (int i = 0; i < data_node_num; i++)
        {
            Dnode_ *dnode_ = new Dnode_();
            if (array_sizes[i] == 1)
            {
                assert(begin_addr == keys.size() - 1);
                // add the only key into last node
                DEBUG_THIS("[ROOT]: init with single record data node" );
                uint8_t loc = this->worker_num -1;
                groups[i - 1].second->insert(keys[begin_addr], vals[begin_addr], loc, false);
                begin_addr ++;
                data_node_num --;
                break;
            }
            dnode_->init(keys, vals, init_expansion, init_epsilon, root_rmi[i], array_sizes[i], begin_addr, init_read_epsilon, worker_num);
            assert(root_rmi[i].key == keys[begin_addr]);
            groups[i].first = keys[begin_addr];
            groups[i].second = dnode_;
            begin_addr += array_sizes[i];
        }
        assert(begin_addr == key_num);
        // Build upper levels
        train_rmi_xindex();
        DEBUG_THIS("[ROOT]: End root initialization!"
                  << "data node number is " << data_node_num );
    }

    /*using the RMI model and then exponential search*/
    template <typename K, typename Val>
    typename Root<K, Val>::Dnode_ *
    Root<K, Val>::locate_group_3(const K &key, size_t &begin_node, bool &link_node)
    {
        size_t group_i = predict(key);
        group_i = group_i > (int)data_node_num - 1 ? data_node_num - 1 : group_i;
        group_i = group_i < 0 ? 0 : group_i;
        if (!groups)
        {
            begin_node = EMPTY_KEY;
            return nullptr;
        }
        // exponential search
        int begin_group_i, end_group_i;
        if (groups[group_i].first <= key)
        {
            size_t step = 1;
            begin_group_i = group_i;
            end_group_i = begin_group_i + step;
            while (end_group_i < (int)data_node_num && groups[end_group_i].first <= key)
            {
                step = step << 1;
                begin_group_i = end_group_i;
                end_group_i = begin_group_i + step;
            } // after this while loop, end_group_i might be >= group_n
            if (end_group_i > (int)data_node_num - 1)
            {
                end_group_i = data_node_num - 1;
            }
        }
        else
        {
            size_t step = 1;
            end_group_i = group_i;
            begin_group_i = end_group_i - step;
            while (begin_group_i >= 0 && groups[begin_group_i].first > key)
            {
                step = step << 1;
                end_group_i = begin_group_i;
                begin_group_i = end_group_i - step;
            } // after this while loop, begin_group_i might be < 0
            if (begin_group_i < 0)
            {
                begin_group_i = -1;
            }
        }
        while (end_group_i != begin_group_i)
        {
            int mid = (end_group_i + begin_group_i + 2) / 2;
            if (groups[mid].first <= key)
            {
                begin_group_i = mid;
            }
            else
            {
                end_group_i = mid - 1;
            }
        }
        group_i = end_group_i < 0 ? 0 : end_group_i;
        Dnode_ *real_node = groups[group_i].second;
        link_node = false;
        while ((real_node->next != nullptr) && (real_node->next != 0))
        {
            link_node = true;
            Dnode_ *tmp = real_node->next;
            if (tmp->min_key <= key)
            {
                real_node = tmp;
            }
            else
            {
                break;
            }
        }
        begin_node = group_i;
        return real_node;
    }



    // different operations
    /*@query:
    get the value of the given key
    if successfully get the value, return ture;
    or return false.
    */
    template <typename K, typename Val>
    inline result_t Root<K, Val>::query(const K &key, Val &val, uint8_t & worker_id, bool countered)
    {
        // fisrt locate the data node
        size_t node_num = -1;
        bool linked = false;
        result_t ans_1;
        Dnode_ * search_node;
        if(data_node_num == 1){
            search_node = groups[0].second;
            node_num = 0;
        }else{
            search_node = locate_group_3(key, node_num, linked);
        }
        
        if(linked){
            countered = false;
        }
        assert(search_node != nullptr);
        //if the search node is in the sync stage
        if(groups[node_num].second->sync_phase == true){
            val_ptr_t ptr_to_val = nullptr;
            ans_1 = search_node->query(key, val, ptr_to_val, worker_id);
            if(ans_1 != result_t::ok){
                //continue to search the shadow node
                Val tmp_result = val;
                Dnode_ * shadow_node = groups[node_num].second->training_node;
                countered = true;
                result_t ans_2 = shadow_node->query_shadownode(key, val, worker_id, countered);
                //the key does not exist
                if(ans_1 == result_t::failed && ans_2 == result_t::failed){
                    return result_t::failed;
                }
                //the key just in the old data node
                if(ans_1 == result_t::failed && ans_2 == result_t::ok){
                    return ans_2;
                }
                //key exist in the new data node, but does not exist in the old data node. some thing wrong
                if(ans_1 == result_t::retry && ans_2 == result_t::failed){
                    DEBUG_THIS("[ERROR]: The sync is wrong!");
                }
                //the key exist in the old and new 
                if(ans_1 == result_t::retry && ans_2 == result_t::ok){
                    //determint wether the value in the new node is the latest version
                    //has been changed
                    Val old_version = *ptr_to_val;
                    if((old_version & SHADOW_MASK) == 0){
                        //update the new data node
                        Val new_val = (val | SHADOW_MASK);
                        *ptr_to_val = new_val;
                        return result_t::ok; 
                    }else{
                        //return the latest version of value
                        val = old_version;
                        return result_t::ok;
                    }                    
                }
            }
            return ans_1;
        }
        //execute the normal way
        ans_1 = search_node->query(key, val, worker_id,countered);
        return ans_1;
    }

    /*@update:
    update the value of the given key
    if successfully get the value, return ture;
    or return false.
    */
    template <typename K, typename Val>
    inline result_t Root<K, Val>::update(const K &key, const Val &val, uint8_t & worker_id, bool countered)
    {
        // fisrt locate the data node
        size_t node_num = -1;
        bool linked = false;
        Dnode_ *search_node = locate_group_3(key, node_num, linked);
        if(linked){
            countered = false;
        }
        if(groups[node_num].second->sync_phase){
            Val new_val = val | SHADOW_MASK;
            return search_node->insert(key, new_val, worker_id, countered);
        }else{
            return search_node->update(key, val, worker_id, countered);
        }
    }

    /*@remove:
    remove the given key value pair
    if successfully get the value, return ture;
    or return false.
    */
    template <typename K, typename Val>
    inline result_t Root<K, Val>::remove(const K &key, uint8_t & worker_id, bool countered)
    {
        // fisrt locate the data node
        size_t node_num = -1;
        bool linked = false;
        result_t ans_1;
        Dnode_ *search_node = locate_group_3(key, node_num, linked);
        if(linked){
            countered = false;
        }
        if(groups[node_num].second->sync_phase){
            Val new_val = SHADOW_MASK;
            return search_node->insert(key, new_val, worker_id, countered);
        }else{
            return search_node->remove(key, worker_id, countered);
        }
    }

    /*@scan:
    retrieve the n smallest keys larger than begin.
    */    
    template <typename K, typename Val>
    inline size_t Root<K,Val>::scan(const K &begin, const size_t n,
        std::vector<std::pair<K, Val>> &result, bool countered){
        int remaining = n;
        result.clear();
        result.reserve(2*n);
        K next_begin = begin;
        K lastest_DN_pivot = 0;
        
        int data_node_i;
        size_t node_num = -1;
        bool linked = false;
        result_t ans_1;
        Dnode_ *search_node = locate_group_3(begin, node_num, linked);
        if(linked){
            countered = false;
        }
        if(!(begin >= search_node->min_key)){
            return 0;
        }
        size_t loc_i = node_num;
        while(remaining > 0){
            size_t done = search_node->scan(begin,remaining,result, countered);
            remaining -= done;
            if(remaining == 0){
                return n;
            }
            loc_i++; 
            if(loc_i >= this->data_node_num){
                break;
            }
            if(search_node->next != nullptr && search_node->next != 0){
                search_node = search_node->next;
            }else{
                search_node = groups[loc_i].second;
            }
        }
        std::sort(result.begin(), result.end(),[](std::pair<K, Val>& a, std::pair<K, Val>& b){
            return a.first < b.first; 
        });
        result.resize(n);
        return n - remaining;
    }


    template <typename K, typename Val>
    inline result_t Root<K, Val>::insert(const K &key, const Val &val, uint8_t & worker_id, bool countered)
    {
        // fisrt locate the data node
        size_t node_num = -1;
        bool linked = false;
        Dnode_ *search_node = locate_group_3(key, node_num, linked);
        if(linked){
            countered = false;
        }
        if(groups[node_num].second->sync_phase){
            Val new_val = val | SHADOW_MASK;
            return search_node->insert(key, new_val, worker_id, countered);
        }else{
            return search_node->insert(key, val, worker_id, countered);
        }
        
    }

    /*Structual modification operation*/
    template <typename K, typename Val>
    void *Root<K, Val>::do_SMO(void *args, worksteal *work_queues)
    {
        // update the metadata
        volatile bool &should_update_array = ((BGInfo *)args)->should_update_array;
        std::atomic<bool> &started = ((BGInfo *)args)->started;
        std::atomic<bool> &finished = ((BGInfo *)args)->finished;
        size_t bg_i = ((BGInfo *)args)->bg_i;
        size_t bg_num = ((BGInfo *)args)->bg_n;
        volatile bool &running = ((BGInfo *)args)->running;
        uint64_t sleep_time = ((BGInfo *)args)->sleep_time;
        worksteal &work_queue = work_queues[bg_i];
        // check each data node and estimate the performance
        while (running)
        {
            if (started)
            {
                // periodically check each data node and do the adjustment
                started = false;
                Root &root = **(Root *volatile *)(((BGInfo *)args)->root_ptr);
                // for analysis
                size_t retraning_time = 0, node_split = 0, node_merge = 0, node_expand = 0;
                while (!work_queue.empty())
                {
                    Dnode_ *volatile *Data_node;
                    // get the first item in the queue
                    std::optional<Dnode_ *volatile *> item = work_queue.pop();
                    if (item.has_value())
                    {
                        Data_node = item.value();
                    }
                    else
                    {
                        continue;
                    }
                    if(root.execute_smo(node_split,node_merge,node_expand,Data_node,root)){
                        should_update_array = true;
                    }
                }
                // steal threads from another queue
                if (bg_i < bg_num >> 1)
                {
                    for (int j = bg_num - 1; j >= 0; j--)
                    {
                        if (j == bg_i)
                        {
                            continue;
                        }
                        while (!work_queues[j].empty())
                        {
                            Dnode_ *volatile *Data_node;

                            // get the first item in the queue
                            std::optional<Dnode_ *volatile *> item = work_queue.steal();
                            if (item.has_value())
                            {
                                Data_node = item.value();
                                if(root.execute_smo(node_split,node_merge,node_expand,Data_node,root)){
                                    should_update_array = true;
                                }
                            }
                            else
                            {
                                continue;
                            }
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < bg_num; j++)
                    {
                        if (j == bg_i)
                        {
                            continue;
                        }
                        while (!work_queues[j].empty())
                        {
                            Dnode_ *volatile *Data_node;

                            // get the first item in the queue
                            std::optional<Dnode_ *volatile *> item = work_queue.steal();
                            if (item.has_value())
                            {
                                Data_node = item.value();
                                if(root.execute_smo(node_split,node_merge,node_expand,Data_node,root)){
                                    should_update_array = true;
                                }
                            }
                            else
                            {
                                continue;
                            }
                        }
                    }
                }
                finished = true;
                if(node_split || node_expand || node_merge){
                    DEBUG_THIS("[BG "<< bg_i <<"] [structure update] whloe_node_split_n: " << node_split);
                    DEBUG_THIS("[BG "<< bg_i <<"] [structure update] node_expand_n: " << node_expand);
                    DEBUG_THIS("[BG "<< bg_i << "] [structure update] node_merge: " << node_merge);
                }
                
            }
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time/10));
    }

template <typename K, typename Val>
bool Root<K, Val>::execute_smo(size_t &split, size_t & merge, size_t &expand, Dnode_ * volatile* Data_node,Root &root)
    {
        bool should_split_whole_node = false;
        bool should_merge_node = false;
        bool should_expand_node = false;

        root.estimate_overhead_dynamically(should_split_whole_node, should_merge_node, should_expand_node, *Data_node);
        // do SMO operation and also need to determine the configuration
        Dnode_ *old_data_node = (*Data_node);
        assert(!((*Data_node)->next));
        if (should_split_whole_node)
        {
            tmp_log<K, Val> * new_write_log = new tmp_log<K, Val>(1);
            old_data_node->retraining_phase = true;
             __atomic_exchange_n(&(old_data_node->write_log), new_write_log, __ATOMIC_SEQ_CST);
             size_t loop_num =0;
            while(old_data_node->before_retrain_counter.load() != 0){
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                loop_num++;
                if(loop_num > 100){
                    std::cout << "loop much counter " << old_data_node->before_retrain_counter.load() << std::endl;
                }
                break;
            }
            Dnode_ *new_Dnode = old_data_node->split_data_node();
            __atomic_exchange_n(Data_node, new_Dnode, __ATOMIC_SEQ_CST);
            loop_num =0;
            while(old_data_node->during_retrain_counter.load() != 0){
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                loop_num++;
                if(loop_num > 100){
                    std::cout << "loop much counter " << old_data_node->during_retrain_counter.load() << std::endl;
                }
                break;
            }

            // update the items in log
            Dnode_ *tmp_one = new_Dnode;
            Dnode_ *prev = nullptr;
            tmp_log<K, Val> *tmp_buf = new_Dnode->write_log;
            assert(tmp_buf == new_write_log);
            //from the tail update the log
            while(1){
                K key_tmp;
                Val val_tmp;
                if(tmp_buf->pop_back(key_tmp,val_tmp)){
                    //need to traverse and update the new data nodes one by one
                    //first find the tmp_node
                    tmp_one = new_Dnode;
                    while(1){
                        if(key_tmp >= tmp_one->min_key && tmp_one->next == nullptr){
                            tmp_one->sync_log(key_tmp, val_tmp);
                            break;
                        }
                        if(key_tmp >= tmp_one->min_key && key_tmp < tmp_one->next->min_key){
                            tmp_one->sync_log(key_tmp, val_tmp);
                            break;
                        }
                        prev = tmp_one;
                        tmp_one = tmp_one->next;
                        if(tmp_one == nullptr){
                            DEBUG_THIS( "[sync wrong!] the key is " << key_tmp << " the last node key is " << prev->min_key);
                        }
                    }
                }else{
                    break;
                }
            }

            //@wait for all clients access the shadow data node
            //update the stage
            new_Dnode->sync_phase = false;
            loop_num =0;
            while(old_data_node->before_retrain_counter.load() != 0){
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                loop_num++;
                if(loop_num > 100){
                    std::cout << "loop much counter " << old_data_node->before_retrain_counter.load() << std::endl;
                }
                break;
            }          
            new_Dnode->training_node= nullptr;
            new_Dnode->write_log = nullptr;
            split++;
            old_data_node->free_data();
            old_data_node->free_log();
            if(new_Dnode->next){
                return true;
            }else{
                return false;
            }
            
        }
        else if (should_merge_node)
        {

            return true;
        }
        else if (should_expand_node)
        {
            // do expansion and compact
            tmp_log<K, Val> * new_write_log = new tmp_log<K, Val>(1);
            old_data_node->retraining_phase = true;
            __atomic_exchange_n(&(old_data_node->write_log), new_write_log, __ATOMIC_SEQ_CST);
            size_t loop_num =0;
            while(old_data_node->before_retrain_counter.load() != 0){
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            loop_num++;
            if(loop_num > 100){
                std::cout << "loop much counter " << old_data_node->before_retrain_counter.load() << std::endl;
            }
            break;
            }
            Dnode_ *new_data_node = old_data_node->compact_phase();
                __atomic_exchange_n(Data_node, new_data_node, __ATOMIC_SEQ_CST);
            loop_num =0;
            while(old_data_node->during_retrain_counter.load() != 0){
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                loop_num++;
                if(loop_num > 100){
                    std::cout << "loop much counter " << old_data_node->during_retrain_counter.load() << std::endl;
                }
                break;
            }

            K key_tmp;
            Val val_tmp;
            while (new_data_node->write_log->pop_back(key_tmp,val_tmp))
            {
                if(new_data_node->sync_log(key_tmp, val_tmp)){
                }
            }     
            new_data_node->sync_phase = false;
            loop_num =0;
            while(old_data_node->before_retrain_counter.load() != 0){
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                loop_num++;
                if(loop_num > 100){
                    std::cout << "loop much counter " << old_data_node->before_retrain_counter.load() << std::endl;
                }
                break;
            }          
            new_data_node->training_node= nullptr;
            new_data_node->write_log = nullptr;
            
            expand ++;
            old_data_node->free_data();
            old_data_node->free_log();
            return false;

        }else{
            old_data_node->new_retrained = true;
        }
        return false;
    }
 
    template <typename K, typename Val>
    void Root<K, Val>::estimate_overhead_dynamically(bool &split, bool & merge, bool &expand, Dnode_ * e_Dnode)
    {
        if(e_Dnode == nullptr){
            return;
        }
        double write_ratio = 0;
        uint64_t write_small = 0;
        uint64_t write_half = 0;
        uint64_t read_small = 0;
        uint64_t read_half = 0;
        uint64_t read_buffer = 0;
        uint64_t buffer_size = 0;
        uint64_t * read_counter = e_Dnode->counter_r;
        uint64_t * write_counter = e_Dnode->counter_w;
        uint64_t node_read_eplision = e_Dnode->read_eplision_;
        size_t his_write = e_Dnode->history_amount;
        size_t his_w_1 = e_Dnode->history_w_1;
        size_t his_w_2 = e_Dnode->history_w_2;
        size_t his_r_1 = e_Dnode->history_r_1;
        size_t his_r_2 = e_Dnode->history_r_2;
        size_t his_r_3 = e_Dnode->history_r_3;
        for(int j = 0; j < this->worker_num; j++){
            write_small += write_counter[3*j];
            write_half += write_counter[3*j+1];
            buffer_size += write_counter[3*j+2];
            read_small += read_counter[3*j];
            read_half += read_counter[3*j+1];
            read_buffer += read_counter[3*j+2];
        }
        size_t read_amount = read_small + read_half + read_buffer - his_r_1 - his_r_2 - his_r_3;
        size_t write_amount = write_small + write_half + buffer_size - his_write;
        if ((write_amount + read_amount) < OP_NUM)
        {
            if(!e_Dnode->cold){
                e_Dnode->cold = true;
            }
            return ;
        }
        else if(write_amount == 0){
            return ;
        }else{
            write_ratio = (double)(write_amount) / (double)(write_amount + read_amount);
            e_Dnode->cold = false;
        } 
        uint64_t init_buffer = e_Dnode->init_buffer_size;
        uint64_t init_size_in =  e_Dnode->init_amount();
        uint64_t array_size_in = e_Dnode->array_size();  
        e_Dnode->history_amount += write_amount;
        e_Dnode->history_w_1 = write_half;
        e_Dnode->history_w_2 = buffer_size;
        e_Dnode->history_r_1 = read_small;
        e_Dnode->history_r_2 = read_half;
        e_Dnode->history_r_3 = read_buffer;
        if(write_ratio >= 0.7){
            e_Dnode->write_intensive = true;
            e_Dnode->read_intensive = false;
        }else if(write_ratio <= 0.3){
            e_Dnode->read_intensive = true;
            e_Dnode->write_intensive = false;
        }else{
            e_Dnode->read_intensive = false;
            e_Dnode->write_intensive = false;
        }

        double read_a = 0;
        double read_b = 0;
        if(read_amount != 0){
            read_a = (double)read_half - his_r_2/(double)read_amount;
            read_b = (double)read_buffer - his_r_3/(double)read_amount;
        }
        double hit_ratio = 0;
        double write_large = 0;
        if(write_amount > 0){
            hit_ratio = (double)(buffer_size - his_w_2) / (double)(write_amount );
            write_large = (double)(write_half - his_w_1) / (double)(write_amount );
        }
        double read_performance = (2*read_a+4*read_b);
        double write_performance =   (6 * write_large + 13 * hit_ratio);
        double performance_degradation = (read_performance * (1 - write_ratio) +  write_performance * write_ratio)/(1 + 2 * write_ratio);
        double fill_ratio = (double)(init_size_in + write_amount + his_write - init_buffer - buffer_size) / (double)array_size_in;
        if(e_Dnode->cold){
            if(fill_ratio < FILL_COLD){
                expand = true;
                return;
            }
            return;
        }
        //decide to perform which SMO
        if(performance_degradation >= DEGRADATION){
            if(e_Dnode->zoomed_level >= LARGE_ZOOMED_LEVEL &&  fill_ratio < FILL_LARGE_R){
                split = true;
                return;
            }
            if(fill_ratio < 0.6){
            }else{
                expand = true;
                return;
            }
            
        }
        if(write_ratio >= 0.7){
            double insert_increment = ((write_amount) *(1 - hit_ratio));
            double fill_ratio_incre = ((double)(insert_increment) / (double)array_size_in);

            if((fill_ratio_incre  +  fill_ratio >=  (double)FUTURE_FILL) ){
                if(e_Dnode->zoomed_level >= LARGE_ZOOMED_LEVEL){
                    split = true;
                }else{
                    expand = true;
                }
                return ;
            }
        }
    }


    /*
    check each data node for newly added data node, and update the root node
    */
    template <typename K, typename Val>
    Root<K, Val> *Root<K, Val>::create_new_root()
    {
        Root *new_root = new Root();
        // create the new root
        new_root->init_epsilon = init_epsilon;
        new_root->init_expansion = init_expansion;
        new_root->init_read_epsilon = init_read_epsilon;
        new_root->worker_num = worker_num;
        size_t new_data_node_num = 0;
        for (size_t Dnode_po = 0; Dnode_po < data_node_num; Dnode_po++)
        {
            Dnode_ *data_node = groups[Dnode_po].second;
            while (data_node)
            {
                new_data_node_num++;
                data_node = data_node->next;
            }
        }
        new_root->groups = std::make_unique<std::pair<K, Dnode_ *volatile>[]>(new_data_node_num);
        new_root->data_node_num = new_data_node_num;
        // copy the content from the old group node
        size_t data_node_i = 0;
        for (size_t Dnode_po = 0; Dnode_po < data_node_num; Dnode_po++)
        {
            Dnode_ *data_node = groups[Dnode_po].second;
            while (data_node)
            {
                new_root->groups[data_node_i].first = data_node->min_key;
                new_root->groups[data_node_i].second = data_node;
                data_node_i++;
                data_node = data_node->next;
            }
        }
        new_root->rmi_2nd_stage_model_n = rmi_2nd_stage_model_n;
        new_root->train_rmi_xindex();

        return new_root;
    }
    /*When get the new node, update all the next pointer to nullptr*/
    template <typename K, typename Val>
    void Root<K, Val>::clear_next()
    {
        for (int i = 0; i < data_node_num; i++)
        {
            groups[i].second->next = nullptr;
            groups[i].second->before_retrain_counter = 0;
        }
    }

    /*XIndex RMI*/
    template <typename K, typename Val>
    void Root<K, Val>::train_rmi_xindex()
    {
        size_t max_model_n = 10 * 1024*1024/sizeof(linear_model_t);
        size_t max_trial_n = 10;
        size_t model_n_trial = rmi_2nd_stage_model_n;
        if (model_n_trial == 0)
        {
            max_trial_n = 100;
            const size_t group_n_per_model_per_rmi_error_experience_factor = 4;
            model_n_trial = std::min(
                max_model_n,        // do not exceed memory constraint
                std::max((size_t)1, // do not decrease to zero
                         (size_t)(data_node_num / root_error_bound /
                                  group_n_per_model_per_rmi_error_experience_factor)));
        }
        train_rmi_xindex_2(model_n_trial, skewd_workloads);
        size_t model_n_trial_prev_prev = 0;
        size_t model_n_trial_prev = model_n_trial;

        size_t trial_i = 0;
        double mean_error = 0;
        for (; trial_i < max_trial_n; trial_i++)
        {
            std::vector<double> errors;
            for (size_t group_i = 0; group_i < data_node_num; group_i++)
            {
                errors.push_back(
                    std::abs((double)group_i - predict(groups[group_i].first)) + 1);
            }
            mean_error =
                std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();

            if (mean_error > root_error_bound)
            {
                if (rmi_2nd_stage_model_n == max_model_n)
                {
                    break;
                }
                model_n_trial = std::min(
                    max_model_n,                        // do not exceed memory constraint
                    std::max(rmi_2nd_stage_model_n + 1, // but at least increase by 1
                             (size_t)(rmi_2nd_stage_model_n * mean_error /
                                      root_error_bound)));
            }
            else if (mean_error < root_error_bound / 2)
            {
                if (rmi_2nd_stage_model_n == 1)
                {
                    break;
                }
                model_n_trial = std::max(
                    (size_t)1,                          // do not decrease to zero
                    std::min(rmi_2nd_stage_model_n - 1, // but at least decrease by 1
                             (size_t)(rmi_2nd_stage_model_n * mean_error /
                                      (root_error_bound / 2))));
            }
            else
            {
                break;
            }
            skewd_workloads = true;
            train_rmi_xindex_2(model_n_trial, skewd_workloads);
            if (model_n_trial == model_n_trial_prev_prev)
            {
                break;
            }
            model_n_trial_prev_prev = model_n_trial_prev;
            model_n_trial_prev = model_n_trial;
        }

        DEBUG_THIS("--- [root] final rmi size: "
                   << rmi_2nd_stage_model_n << " (error=" << mean_error << "), after "
                   << trial_i << " trial(s)");
    }

    template <typename K, typename Val>
    inline size_t Root<K, Val>::predict(const K &key)
    {
        size_t pos_pred;
        if(skewd_workloads){
            if(key >= rmi_1st_stage_backup.begin_addr){
                pos_pred = rmi_1st_stage_backup.predict_rmi(key);
            }else{
                pos_pred = rmi_1st_stage.predict_rmi(key);
            }
        }else{
            pos_pred = rmi_1st_stage.predict_rmi(key);
        } 
        size_t next_stage_model_i = pick_next_stage_model(pos_pred);
        return rmi_2nd_stage[next_stage_model_i].predict_rmi(key);
    }

    template <typename K, typename Val>
    inline void Root<K, Val>::train_rmi_xindex_2(size_t rmi_2nd_stage_model_n, bool skew)
    {
        this->rmi_2nd_stage_model_n = rmi_2nd_stage_model_n;
        delete[] rmi_2nd_stage;
        rmi_2nd_stage = new linear_model_t[rmi_2nd_stage_model_n]();
        // train 1st stage, if skew, using two linear models 
        std::vector<K> keys;
        std::vector<size_t> positions;
        size_t key_loc = 0;

        for (size_t group_i = 0; group_i < data_node_num; group_i++)
        {
            keys.push_back(groups[group_i].first);
            positions.push_back(group_i);
            if(skew && group_i == data_node_num/2 - 1){
                break;
            }
        }
        rmi_1st_stage.prepare(keys, positions);
        if(skew){
            for (size_t group_i = data_node_num/2; group_i < data_node_num; group_i++)
            {
                if(key_loc == data_node_num/2){
                    keys.push_back(groups[group_i].first);
                    positions.push_back(group_i);
                    break;
                }
                keys[key_loc] = (groups[group_i].first);
                positions[key_loc] = (group_i);
                key_loc ++;
            }
            rmi_1st_stage_backup.prepare(keys, positions);
        }

        // train 2nd stage
        std::vector<std::vector<K>> keys_dispatched(rmi_2nd_stage_model_n);
        std::vector<std::vector<size_t>> positions_dispatched(rmi_2nd_stage_model_n);

        for (size_t key_i = 0; key_i < data_node_num; ++key_i)
        {
            size_t group_i_pred;
            if(!skew){
                group_i_pred = rmi_1st_stage.predict_rmi(groups[key_i].first);
            }else{
                if(key_i >= data_node_num/2){
                    group_i_pred = rmi_1st_stage_backup.predict_rmi(groups[key_i].first);
                }
            }
            size_t next_stage_model_i = pick_next_stage_model(group_i_pred);
            keys_dispatched[next_stage_model_i].push_back(groups[key_loc].first);
            positions_dispatched[next_stage_model_i].push_back(key_i);
        }
        int last_valid_model = -1;
        for (size_t model_i = 0; model_i < rmi_2nd_stage_model_n; ++model_i)
        {
            std::vector<K> &new_keys = keys_dispatched[model_i];
            std::vector<size_t> &new_positions = positions_dispatched[model_i];
            //label the empty models
            rmi_2nd_stage[model_i].prepare(new_keys, new_positions);
            if(new_keys.size() == 0){
                 rmi_2nd_stage[model_i].begin_addr = last_valid_model;
            }else{
                last_valid_model = model_i;
            }
        }
    }

    template <class K, class Val>
    size_t Root<K, Val>::pick_next_stage_model(size_t group_i_pred)
    {
        size_t second_stage_model_i;
        second_stage_model_i = group_i_pred * rmi_2nd_stage_model_n / data_node_num;

        if (second_stage_model_i >= rmi_2nd_stage_model_n)
        {
            second_stage_model_i = rmi_2nd_stage_model_n - 1;
        }

        return second_stage_model_i;
    }

}
