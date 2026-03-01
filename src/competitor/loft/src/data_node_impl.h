#pragma once 
#include "data_node.h"
#include <cstdio>

namespace loft{
template<typename K, typename Val>
Dnode<K,Val>::Dnode() {}

template<typename K, typename Val>
Dnode<K,Val>::~Dnode() {}

template<typename K, typename Val>
void Dnode<K,Val>::init(const std::vector<K> & key_begin,const std::vector<Val> & val_begin, double expansion,  size_t epsilon, segment seg, uint64_t array_size, size_t begin_addr, size_t read_epsilon, size_t worker_num){
    //first get the useful information     
    this->expansion_ = expansion;
    this->epsilon_ = epsilon;
    this->min_key = key_begin[begin_addr];
    this->init_size = array_size;
    this->worker_num = worker_num;
    if(init_size == 0){
        // std::cout << "init size is 0" << std::endl;
        return ;
    }
    this-> read_eplision_ = read_epsilon;
    this->array_size_ = ceil(double(array_size + 2*epsilon + 4) * expansion);
    this->keys = new K[this->array_size_];
    this->vals = new Val[this->array_size_];
    double tmp1 = seg.slope * expansion;
    int tmp2 = 0;
    this->model = new LinearRegressionModel<K>(tmp1, tmp2, this->min_key);
    size_t zoomed_amount = this->array_size_ / ZOOM_SHARE + 1;
    this->zoomed_keys = (K**)calloc(zoomed_amount, sizeof(K*));
    this->zoomed_vals = (Val**)calloc(zoomed_amount, sizeof(Val*));
    this->counter_r = (size_t*)calloc(this->worker_num *3, sizeof(size_t));
    this->counter_w = (size_t*)calloc(this->worker_num *3, sizeof(size_t));
    size_t bias = 0;
    int64_t relative_po = 0;
    int64_t pivot = -1;
    typename std::vector<K>::const_iterator iter_key = key_begin.begin() + begin_addr;
    typename std::vector<Val>::const_iterator iter_val = val_begin.begin() + begin_addr;
    size_t init_buffer = 0;
    //allocate the exact place & place the records, we need to go through
    for(int i = 0; i < array_size; i++){
        //just skip the redundant keys
        relative_po = this->model->predict(*iter_key);
        if(relative_po < array_size_ && relative_po > pivot){
            //just insert into the predicted posistion
            this->keys[relative_po] = *iter_key;
            this->vals[relative_po] = *iter_val;
            iter_key++;
            iter_val++;
        }else if(pivot < array_size_ - 1 && (pivot >= relative_po && (pivot - relative_po + 1) < read_eplision_)){
            bias += (pivot + 1 - relative_po); 
            relative_po = pivot +1;
            this->keys[relative_po] = *iter_key;
            this->vals[relative_po] = *iter_val;
            iter_key++;
            iter_val++;
        }else{
                bias += read_eplision_;
                init_buffer ++;
                bool entered = false;
                //insert into buffer
                if(relative_po >= this->array_size_){
                    relative_po = this->array_size_;
                }
                int loc = relative_po/ZOOM_SHARE;
                double full_lo = this->model->predict(*iter_key);
                if(full_lo >= this->array_size_){
                    full_lo = this->array_size_;
                }
                int begin_addr_i =  (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
                K** tmp_zoomed_kslot = &zoomed_keys[loc];
                Val** tmp_zoomed_vslot = &zoomed_vals[loc];
                int current_zoomed_level = 0;
                //add the buffer size
                this->zoomed = true;
                while(1){
                    current_zoomed_level += 1;
                    if(*tmp_zoomed_kslot== 0){
                        //try to allocate the key region_slot, need to guarantee that there is only one
                        K * new_zoomed_kslot = new K[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 2] ();
                        *tmp_zoomed_kslot = new_zoomed_kslot;
                    }
                    if(*tmp_zoomed_vslot == 0){
                        //other thread allocate the key region but not allocate the value region
                        Val * new_zoomed_vslot = new Val[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 2];
                        new_zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1] = 0;
                        *tmp_zoomed_vslot = new_zoomed_vslot;
                    }
                    if(current_zoomed_level > zoomed_level){
                        zoomed_level = current_zoomed_level;
                    }
                    //make sure that both Key and value slot exist
                    K * k_slot = *tmp_zoomed_kslot;
                    Val * v_slot = *tmp_zoomed_vslot;
                    for(int j = 0; j < read_eplision_; j++){
                        uint64_t empty = EMPTY_KEY;
                        // if(*(uint64_t *)&(k_slot[begin_addr+j]) == empty){
                        if((k_slot[begin_addr_i+j]) == empty){
                            k_slot[begin_addr_i+j] = *iter_key;
                            v_slot[j+begin_addr_i] = *iter_val;
                            entered = true;
                            break;
                        }


                    }
                    if(entered){
                        break;
                    }
                    //is full, try to find the next level
                    tmp_zoomed_kslot = (K **)(&k_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
                    tmp_zoomed_vslot = (Val **)(&v_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
                }
                iter_key++;
                iter_val++;
        }
        //update the bitmap
        size_t array_loc = pivot+1;
        while(array_loc < relative_po){
            *((uint64_t*)&keys[array_loc]) = EMPTY_KEY;
            array_loc++;
        }    
        if(relative_po > pivot){
            pivot = relative_po;
        }
        
    }
    //to fit the empty place from pivot to array_size_
    init_buffer_size = init_buffer;
    pivot ++;
    while(pivot < array_size_){
        *((uint64_t*)&keys[pivot]) = EMPTY_KEY;
        pivot++;
    }

}




/*@query:
get the value of the given key
if successfully get the value, return ture; 
or return false.
*/
template<typename K, typename Val>
inline result_t Dnode<K, Val>::query(const K &key, Val &val, uint8_t & worker_id, bool counter){
    bool before_retrain = false;
    bool during_retrain = false;
    //master bg start retraining, but this node has not been retrained  
    K min_key_in_model = this->model->get_pivot();
    
    size_t read_depth = 0;
    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    size_t relative_po = predict_po;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        full_lo = this->array_size_;
    }
    if(key >= min_key_in_model && predict_po < this->array_size_){
        size_t begin_a = predict_po;
        size_t end_a = predict_po + read_eplision_ >= this->array_size_ ? this->array_size_ : predict_po + read_eplision_;
        end_a --;
        if(0){

        }else{
            while(end_a >= begin_a){
                if(keys[begin_a] == key ){
                        if(read_depth >= (read_eplision_/2)){
                            counter_r[worker_id*3+1]++;
                        }else{
                            counter_r[worker_id*3]++;
                        }
                        val = vals[begin_a];
                        return result_t::ok;
                } 
                if(keys[begin_a] == EMPTY_KEY){
                    return result_t::failed;
                }
                read_depth++;
                begin_a ++;
            }
        }

    }
    if(key < min_key_in_model){
        // DEBUG_THIS("[error:] query_1! smaller than the min_key in the model");
        return result_t::failed;
    }

    // continue to check the zoomed part
    if(zoomed){
        counter_r[worker_id*3+2]++;
        //find the owner zoomed area
        int g_zoom = predict_po/ZOOM_SHARE;
        //check the zoomed slot
        K * zoomed_kslot = zoomed_keys[g_zoom];
        Val * zoomed_vslot = zoomed_vals[g_zoom];
        //both exist, and the zoomed slot is visiable
        int current_zoomed_level = 0;
        if(zoomed_kslot != 0 && zoomed_vslot != 0){
            int begin_addr = (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
            while(1){
                for(int j = 0; j < read_eplision_; j++){
                    if(zoomed_kslot[j+begin_addr] == key){
                        val = zoomed_vslot[j+begin_addr];
                        return result_t::ok;
                    }
                    if(zoomed_kslot[j+begin_addr] == EMPTY_KEY){
                        return result_t::failed;
                    }
                }
                if(zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1] != 0){
                    //enter to the next level
                    zoomed_kslot = (K *)*(&zoomed_kslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
                    zoomed_vslot = (Val *)*(&zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
                    // std::cout << "search enter to the next level" << std::endl;
                }else{
                    break;
                }
            }
        }
    }
    return result_t::failed;
}
template<typename K, typename Val>
inline result_t Dnode<K, Val>::query(const K &key, Val &val, val_ptr_t & val_ptr, uint8_t & worker_id){
    K min_key_in_model = this->model->get_pivot();
    
    size_t read_depth = 0;
    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    size_t relative_po = full_lo;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        relative_po = this->array_size_;
        full_lo = this->array_size_;
    }
    if(key >= min_key_in_model && predict_po < this->array_size_){
        size_t begin_a = predict_po;
        size_t end_a = predict_po + read_eplision_ >= this->array_size_ ? this->array_size_ : predict_po + read_eplision_;
        end_a --;
        if(skew){
            int step = end_a - begin_a;
            int loc = find_first_num_avx2(key, &keys[begin_a], step);
            if(loc != -1){
                val = vals[begin_a + loc];
                val_ptr = &(vals[begin_a + loc]);
                return result_t::ok;
            }
            return result_t::failed;
        }else{
            while(end_a >= begin_a){
                if(keys[begin_a] == key ){
                        val = vals[begin_a];
                        val_ptr = &(vals[begin_a]);
                        return result_t::ok;
                } 
                if(keys[begin_a] == EMPTY_KEY){
                    return result_t::failed;
                }
                read_depth++;
                begin_a ++;
            }
        }

    }
    if(key < min_key_in_model){
        DEBUG_THIS("[error:] query_1! smaller than the min_key in the model");
    }

    // continue to check the zoomed part
    if(zoomed){
        counter_r[worker_id*3+2]++;
        //find the owner zoomed area
        int g_zoom = predict_po/read_eplision_;
        //check the zoomed slot
        K * zoomed_kslot = zoomed_keys[g_zoom];
        Val * zoomed_vslot = zoomed_vals[g_zoom];
        //both exist, and the zoomed slot is visiable
        int current_zoomed_level = 0;
        if(zoomed_kslot != 0 && zoomed_vslot != 0){
            int begin_addr = (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
            while(1){
                current_zoomed_level ++;

                for(int j = 0; j < read_eplision_; j++){
                    if(zoomed_kslot[j+begin_addr] == key){
                        val = zoomed_vslot[j+begin_addr];
                        return result_t::ok;
                    }
                    if(zoomed_kslot[j+begin_addr] == EMPTY_KEY){
                        return result_t::failed;
                    }
                }
                if(zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_] != 0){
                    //enter to the next level
                    zoomed_kslot = (K *)*(&zoomed_kslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_]);
                    zoomed_vslot = (Val *)*(&zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_]);
                    // std::cout << "search enter to the next level" << std::endl;
                }else{
                    break;
                }
            }
        }
    }
    return result_t::failed;
}
template<typename K, typename Val>
inline result_t Dnode<K, Val>::query_shadownode(const K &key, Val &val, uint8_t & worker_id, bool counter){
    this->before_retrain_counter.fetch_add(1);
    K min_key_in_model = this->model->get_pivot();
    
    size_t read_depth = 0;
    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        full_lo = this->array_size_;
    }
    size_t relative_po = predict_po;
    if(key >= min_key_in_model && predict_po < this->array_size_){
        size_t begin_a = predict_po;
        size_t end_a = predict_po + read_eplision_ >= this->array_size_ ? this->array_size_ : predict_po + read_eplision_;
        end_a --;
        if(skew){
            int step = end_a - begin_a;
            int loc = find_first_num_avx2(key, &keys[begin_a], step);
            if(loc != -1){
                val = vals[begin_a + loc];
                    this->before_retrain_counter.fetch_sub(1);

                return result_t::ok;
            }

            this->before_retrain_counter.fetch_sub(1);

            return result_t::failed;
        }else{
            while(end_a >= begin_a){
                if(keys[begin_a] == key ){
                        val = vals[begin_a];
                        this->before_retrain_counter.fetch_sub(1);

                        return result_t::ok;
                } 
                if(keys[begin_a] == EMPTY_KEY){
                    this->before_retrain_counter.fetch_sub(1);
                    
                    return result_t::failed;
                }
                read_depth++;
                begin_a ++;
            }
        }

    }
    if(key < min_key_in_model){
        DEBUG_THIS("[error:] query_1! smaller than the min_key in the model");
    }

    // continue to check the zoomed part
    if(zoomed){
        //find the owner zoomed area
        int g_zoom = predict_po/ZOOM_SHARE;
        //check the zoomed slot
        K * zoomed_kslot = zoomed_keys[g_zoom];
        Val * zoomed_vslot = zoomed_vals[g_zoom];
        //both exist, and the zoomed slot is visiable
        int current_zoomed_level = 0;
        if(zoomed_kslot != 0 && zoomed_vslot != 0){
            int begin_addr = (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
            while(1){

                for(int j = 0; j < read_eplision_; j++){
                    if(zoomed_kslot[j+begin_addr] == key){
                        val = zoomed_vslot[j+begin_addr];
                        this->before_retrain_counter.fetch_sub(1);
                        
                        return result_t::ok;
                    }
                    if(zoomed_kslot[j+begin_addr] == EMPTY_KEY){
                        this->before_retrain_counter.fetch_sub(1);
                        return result_t::failed;
                    }
                }
                if(zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1] != 0){
                    //enter to the next level
                    zoomed_kslot = (K *)*(&zoomed_kslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
                    zoomed_vslot = (Val *)*(&zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
                    // std::cout << "search enter to the next level" << std::endl;
                }else{
                    break;
                }
            }
        }
    }

    this->before_retrain_counter.fetch_sub(1);
    return result_t::failed;
}
template<typename K, typename Val>
inline result_t Dnode<K, Val>::update(const K &key, const Val &val, uint8_t & worker_id, bool counter){
    bool before_retrain = false;
    bool during_retrain = false;
    //master bg start retraining, but this node has not been retrained  
    if(counter && !retraining_phase && !sync_phase && !new_retrained){
        this->before_retrain_counter.fetch_add(1);
        before_retrain = true;
    }
    //this node has been retrained 
    if(!before_retrain && retraining_phase  && this->write_log != nullptr){
        this->during_retrain_counter.fetch_add(1);
        during_retrain = true;
    }
    K min_key_in_model = this->model->get_pivot();
    
    size_t read_depth = 0;
    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        full_lo = this->array_size_;
    }
    size_t relative_po = predict_po;
    if(key >= min_key_in_model && predict_po < this->array_size_){
        size_t begin_a = predict_po;
        size_t end_a = predict_po + read_eplision_ >= this->array_size_ ? this->array_size_ : predict_po + read_eplision_;
        end_a --;
        if(skew){
            int step = end_a - begin_a;
            int loc = find_first_num_avx2(key, &keys[begin_a], step);
            if(loc != -1){
                if(loc >= (read_eplision_/2)){
                    counter_r[worker_id*3+1]++;
                }else{
                    counter_r[worker_id*3]++;
                }
                //needs to insert into the log
                if(during_retrain){
                    if(!(vals[begin_a + loc] & LOG_MASK)){
                        this->write_log->push_back(key,&vals[begin_a + loc]);
                    }
                    vals[begin_a + loc] = val| LOG_MASK;
                }else{
                    vals[begin_a + loc] = val;
                }             
                
                if(before_retrain){
                    this->before_retrain_counter.fetch_sub(1);
                }
                if(during_retrain){
                    this->during_retrain_counter.fetch_sub(1);
                }
                return result_t::ok;
            }
            if(before_retrain){
                this->before_retrain_counter.fetch_sub(1);
            }
            if(during_retrain){
                this->during_retrain_counter.fetch_sub(1);
            }
            return result_t::failed;
        }else{
            while(end_a >= begin_a){
                if(keys[begin_a] == key ){
                        if(read_depth >= (read_eplision_/2)){
                            counter_r[worker_id*3+1]++;
                        }else{
                            counter_r[worker_id*3]++;
                        }
                        if(during_retrain){
                            if(!(vals[begin_a] & LOG_MASK)){
                                this->write_log->push_back(key,&vals[begin_a]);
                            }
                            vals[begin_a] = val| LOG_MASK;
                        }else{
                            vals[begin_a] = val;
                        }
                        
                        if(before_retrain){
                            this->before_retrain_counter.fetch_sub(1);
                        }
                        if(during_retrain){
                            this->during_retrain_counter.fetch_sub(1);
                        }
                        return result_t::ok;
                } 
                if(keys[begin_a] == EMPTY_KEY){
                    if(before_retrain){
                        this->before_retrain_counter.fetch_sub(1);
                    }
                    if(during_retrain){
                        this->during_retrain_counter.fetch_sub(1);
                    }
                    return result_t::failed;
                }
                read_depth++;
                begin_a ++;
            }
        }

    }
    if(key < min_key_in_model){
        DEBUG_THIS("[error:] query_1! smaller than the min_key in the model");
    }

    // continue to check the zoomed part
    if(zoomed){
        counter_r[worker_id*3+2]++;
        //find the owner zoomed area
        int g_zoom = predict_po/read_eplision_;
        //check the zoomed slot
        K * zoomed_kslot = zoomed_keys[g_zoom];
        Val * zoomed_vslot = zoomed_vals[g_zoom];
        //both exist, and the zoomed slot is visiable
        int current_zoomed_level = 0;
        if(zoomed_kslot != 0 && zoomed_vslot != 0){
            int begin_addr = (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
            while(1){
                current_zoomed_level ++;

                for(int j = 0; j < read_eplision_; j++){
                    if(zoomed_kslot[j+begin_addr] == key){
                        if(during_retrain){
                             if(!(zoomed_vslot[j+begin_addr] & LOG_MASK)){
                                this->write_log->push_back(key,&zoomed_vslot[j+begin_addr]);
                            }
                            zoomed_vslot[j+begin_addr] = val| LOG_MASK;                           
                        }else{
                            zoomed_vslot[j+begin_addr] = val;
                        }
                        if(before_retrain){
                            this->before_retrain_counter.fetch_sub(1);
                        }
                        if(during_retrain){
                            this->during_retrain_counter.fetch_sub(1);
                        }
                        return result_t::ok;
                    }
                    if(zoomed_kslot[j+begin_addr] == EMPTY_KEY){
                        if(before_retrain){
                            this->before_retrain_counter.fetch_sub(1);
                        }
                        if(during_retrain){
                            this->during_retrain_counter.fetch_sub(1);
                        }
                        return result_t::failed;
                    }
                }
                if(zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_] != 0){
                    //enter to the next level
                    zoomed_kslot = (K *)*(&zoomed_kslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_]);
                    zoomed_vslot = (Val *)*(&zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_]);
                }else{
                    break;
                }
            }
        }
    }
    if(before_retrain){
        this->before_retrain_counter.fetch_sub(1);
    }
    if(during_retrain){
        this->during_retrain_counter.fetch_sub(1);
    }
    return result_t::failed;
}

template<typename K, typename Val>
inline result_t Dnode<K, Val>::remove(const K &key, uint8_t & worker_id, bool counter){
    bool before_retrain = false;
    bool during_retrain = false;
    //master bg start retraining, but this node has not been retrained  
    if(counter && !retraining_phase && !sync_phase && !new_retrained){
        this->before_retrain_counter.fetch_add(1);
        before_retrain = true;
    }
    //this node has been retrained 
    if(!before_retrain && retraining_phase  && this->write_log != nullptr){
        this->during_retrain_counter.fetch_add(1);
        during_retrain = true;
    }
    K min_key_in_model = this->model->get_pivot();
    
    size_t read_depth = 0;
    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        full_lo = this->array_size_;
    }
    size_t relative_po = predict_po;
    if(key >= min_key_in_model && predict_po < this->array_size_){
        size_t begin_a = predict_po;
        size_t end_a = predict_po + read_eplision_ >= this->array_size_ ? this->array_size_ : predict_po + read_eplision_;
        end_a --;
        if(skew){
            int step = end_a - begin_a;
            int loc = find_first_num_avx2(key, &keys[begin_a], step);
            if(loc != -1){
                if(loc >= (read_eplision_/2)){
                    counter_r[worker_id*3+1]++;
                }else{
                    counter_r[worker_id*3]++;
                }
                //needs to insert into the log
                if(during_retrain){
                    if(!(vals[begin_a + loc] & LOG_MASK)){
                        this->write_log->push_back(key,&vals[begin_a + loc]);
                    }
                    vals[begin_a + loc] = 0| LOG_MASK;
                }else{
                    vals[begin_a + loc] = 0;
                }             
                
                if(before_retrain){
                    this->before_retrain_counter.fetch_sub(1);
                }
                if(during_retrain){
                    this->during_retrain_counter.fetch_sub(1);
                }
                return result_t::ok;
            }
            if(before_retrain){
                this->before_retrain_counter.fetch_sub(1);
            }
            if(during_retrain){
                this->during_retrain_counter.fetch_sub(1);
            }
            return result_t::failed;
        }else{
            while(end_a >= begin_a){
                if(keys[begin_a] == key ){
                        if(read_depth >= (read_eplision_/2)){
                            counter_r[worker_id*3+1]++;
                        }else{
                            counter_r[worker_id*3]++;
                        }
                        if(during_retrain){
                            if(!(vals[begin_a] & LOG_MASK)){
                                this->write_log->push_back(key,&vals[begin_a]);
                            }
                            vals[begin_a] = 0| LOG_MASK;
                        }else{
                            vals[begin_a] = 0;
                        }
                        
                        if(before_retrain){
                            this->before_retrain_counter.fetch_sub(1);
                        }
                        if(during_retrain){
                            this->during_retrain_counter.fetch_sub(1);
                        }
                        return result_t::ok;
                } 
                if(keys[begin_a] == EMPTY_KEY){
                    if(before_retrain){
                        this->before_retrain_counter.fetch_sub(1);
                    }
                    if(during_retrain){
                        this->during_retrain_counter.fetch_sub(1);
                    }
                    return result_t::failed;
                }
                read_depth++;
                begin_a ++;
            }
        }

    }
    if(key < min_key_in_model){
        DEBUG_THIS("[error:] query_1! smaller than the min_key in the model");
    }

    // continue to check the zoomed part
    if(zoomed){
        counter_r[worker_id*3+2]++;
        //find the owner zoomed area
        int g_zoom = predict_po/ZOOM_SHARE;
        //check the zoomed slot
        K * zoomed_kslot = zoomed_keys[g_zoom];
        Val * zoomed_vslot = zoomed_vals[g_zoom];
        //both exist, and the zoomed slot is visiable
        int current_zoomed_level = 0;
        if(zoomed_kslot != 0 && zoomed_vslot != 0){
            int begin_addr = (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
            while(1){
                current_zoomed_level ++;
                for(int j = 0; j < read_eplision_; j++){
                    if(zoomed_kslot[j+begin_addr] == key){
                        if(during_retrain){
                             if(!(zoomed_vslot[j+begin_addr] & LOG_MASK)){
                                this->write_log->push_back(key,&zoomed_vslot[j+begin_addr]);
                            }
                            zoomed_vslot[j+begin_addr] = 0| LOG_MASK;                           
                        }else{
                            zoomed_vslot[j+begin_addr] = 0;
                        }
                        if(before_retrain){
                            this->before_retrain_counter.fetch_sub(1);
                        }
                        if(during_retrain){
                            this->during_retrain_counter.fetch_sub(1);
                        }
                        return result_t::ok;
                    }
                    if(zoomed_kslot[j+begin_addr] == EMPTY_KEY){
                        if(before_retrain){
                            this->before_retrain_counter.fetch_sub(1);
                        }
                        if(during_retrain){
                            this->during_retrain_counter.fetch_sub(1);
                        }
                        return result_t::failed;
                    }
                }
                if(zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_] != 0){
                    //enter to the next level
                    zoomed_kslot = (K *)*(&zoomed_kslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_]);
                    zoomed_vslot = (Val *)*(&zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_]);
                }else{
                    break;
                }
            }
        }
    }
    if(before_retrain){
        this->before_retrain_counter.fetch_sub(1);
    }
    if(during_retrain){
        this->during_retrain_counter.fetch_sub(1);
    }
    return result_t::failed;
}


//need to consider this
template<typename K, typename Val>
inline size_t Dnode<K, Val>::scan(const K &begin, const size_t n,
        std::vector<std::pair<K, Val>> &result, bool counter){
        bool before_retrain = false;
        bool during_retrain = false;
        //master bg start retraining, but this node has not been retrained  
        if(counter && !retraining_phase && !sync_phase && !new_retrained){
            this->before_retrain_counter.fetch_add(1);
            before_retrain = true;
        }
        //this node has been retrained 
        if(!before_retrain && retraining_phase  && this->write_log != nullptr){
            this->during_retrain_counter.fetch_add(1);
            during_retrain = true;
        }
        size_t remaining = n;
        size_t predict_po = 0;
        if(begin >= this->min_key){
           predict_po = model->predict(begin);
        } 
        size_t zoomed_loc;
        size_t largest_zoomed = (this->array_size_ ) / this->read_eplision_;
        if(predict_po >= this->array_size_){
            zoomed_loc = largest_zoomed;
        }else{
            zoomed_loc = predict_po/this->read_eplision_;
        }
        size_t loc = predict_po % this->read_eplision_;
        size_t start_ = predict_po;
        //first check the data array and then check the zoomed bucket 
        while(remaining && predict_po < array_size_){
            if(keys[predict_po] != EMPTY_KEY){
                if(keys[predict_po]>=begin){
                    Val tmp_val;
                    K tmp_key = keys[predict_po];
                    tmp_val = vals[predict_po];
                    std::pair<K, Val> tmp_res = std::make_pair(tmp_key, tmp_val);
                    //sort the exist result
                    result.push_back(tmp_res);
                    remaining --;
                }
            }
            predict_po++;
        }
        size_t end_bucket = (predict_po-1)/this->read_eplision_ + 1;
        if(end_bucket > largest_zoomed){
            end_bucket = largest_zoomed;
        }
        for (int k = zoomed_loc; k <= end_bucket; k++){
            if(zoomed_keys[k] != nullptr && zoomed_vals[k] != nullptr){
                K * K_slot = zoomed_keys[k];
                Val * V_slot = zoomed_vals[k];
                for(int j = 0; j < read_eplision_ * ZOOM_FACTOR; j++){
                    bool jump = false;
                    if(K_slot[j] != EMPTY_KEY){
                        if(K_slot[j] >=begin){
                            Val tmp_val;
                            K tmp_key = K_slot[j];
                            tmp_val = V_slot[j];
                            std::pair<K, Val> tmp_res = std::make_pair(tmp_key, tmp_val);
                            //sort the exist result
                            result.push_back(tmp_res);
                            if(remaining != 0){
                                remaining --;
                            }
                           
                            }
                        }
                    }

                }
        }
            //continue check the following pos and zoomed bucket
            for(int j = 0; j < this->read_eplision_; j++){
                if(j+predict_po >= this->array_size_){
                    break;
                }
                if(keys[j+predict_po] != EMPTY_KEY){
                    if(keys[j] < result.back().first){
                        //insert this to the result
                        K tmp_key = keys[j+predict_po];
                        Val tmp_val = vals[j+predict_po];
                        std::pair<K, Val> tmp_res = std::make_pair(tmp_key, tmp_val);
                        result.push_back(tmp_res);
                        if(remaining != 0){
                                remaining --;
                        }
                    }
                }
            
        }
        if(before_retrain){
            this->before_retrain_counter.fetch_sub(1);
        }
        if(during_retrain){
            this->during_retrain_counter.fetch_sub(1);
        }   
        return n - remaining;
    }

template<typename K, typename Val>
inline result_t Dnode<K, Val>:: insert(const K &key,const Val &val, uint8_t & worker_id, bool counter){
    //one step:first search the data array
    bool before_retrain = false;
    bool during_retrain = false;
    //master bg start retraining, but this node has not been retrained  
    if(counter && !retraining_phase && !sync_phase && !new_retrained){
        this->before_retrain_counter.fetch_add(1);
        before_retrain = true;
    }
    //this node has been retrained 
    if(!before_retrain && retraining_phase  && this->write_log != nullptr){
        this->during_retrain_counter.fetch_add(1);
        during_retrain = true;
    }

    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    K min_key_in_model = this->min_key;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        full_lo = this->array_size_;
    }
    size_t insert_depth = 0;
    for(int k = 0; k < read_eplision_;k++){
        if(predict_po >= array_size_) break;
        uint64_t empty = EMPTY_KEY;
        if(keys[predict_po] == empty){
            if(CAS((uint64_t *)&keys[predict_po], &empty, key)){
                if(insert_depth <= read_eplision_/2 ){
                    counter_w[worker_id*3]++;
                }else{
                    counter_w[worker_id*3+1]++;
                }
                //continue the CAS the value  
                if(during_retrain){
                    Val tmp_val = (val | LOG_MASK);
                    this->write_log->push_back(key, &vals[predict_po]);
                    vals[predict_po] = tmp_val; 
                    during_retrain_counter.fetch_sub(1);
                    return result_t::ok;
                }
                vals[predict_po] = val; 
                if(before_retrain){
                    before_retrain_counter.fetch_sub(1);
                }
                return result_t::ok;
            }
                    //perform update
            if(empty == key){
                Val tmp_val = vals[predict_po];
                if(during_retrain){
                    if(!(tmp_val & LOG_MASK)){
                        this->write_log->push_back(key, &vals[predict_po]);
                    }
                    vals[predict_po] = val | LOG_MASK;
                    during_retrain_counter.fetch_sub(1);
                    return result_t::ok;
                }
                vals[predict_po] = val;
                if(before_retrain){
                    before_retrain_counter.fetch_sub(1);
                }
                return result_t::ok;
            }
        }
        //perform update
        if(keys[predict_po] == key){
            Val tmp_val = vals[predict_po];
            if(during_retrain){
                if(!(tmp_val & LOG_MASK)){
                    this->write_log->push_back(key, &vals[predict_po]);
                }
                vals[predict_po] = val | LOG_MASK;
                during_retrain_counter.fetch_sub(1);
                return result_t::ok;
            }
            vals[predict_po] = val;
            if(before_retrain){
                before_retrain_counter.fetch_sub(1);
            }
            return result_t::ok;
        }
        predict_po ++;
        insert_depth++;
    }
    //insert to the zoomed area
    if(!zoomed){
        zoomed = true;
    }
    //whether this slot is created
    predict_po = full_lo;
    int loc = predict_po/ZOOM_SHARE;
    int begin_addr = (predict_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)predict_po)*ZOOM_FACTOR;
    this->insert_to_expand_bucket(key, val, loc, begin_addr, worker_id);
    return result_t::ok;
}

template<typename K, typename Val>
inline result_t Dnode<K, Val>::insert_to_expand_bucket(const K &key,const Val &val, size_t loc, size_t predict_po, uint8_t & worker_id){
    //insert to the zoomed area
    if(!zoomed){
        zoomed = true;
    }
    int begin_addr = predict_po;
    K** tmp_zoomed_kslot = &zoomed_keys[loc];
    Val** tmp_zoomed_vslot = &zoomed_vals[loc];
    int current_zoomed_level = 0;
    
    while(1){
        current_zoomed_level += 1;
        if( *tmp_zoomed_kslot == 0){
            //try to allocate the key region_slot, need to guarantee that there is only one
            K * new_zoomed_kslot = (K *)calloc(ZOOM_SHARE * ZOOM_FACTOR +read_eplision_ +2, sizeof(K));
            K * null_ptr = 0;
            if(!CAS(tmp_zoomed_kslot, &null_ptr, new_zoomed_kslot)){
                free(new_zoomed_kslot);
            }
        }
        if(*tmp_zoomed_vslot == 0){
            //other thread allocate the key region but not allocate the value region
            Val * new_zoomed_vslot = (Val *)malloc((ZOOM_SHARE * ZOOM_FACTOR +read_eplision_ +2)*sizeof(Val));
            new_zoomed_vslot[ZOOM_SHARE * ZOOM_FACTOR +read_eplision_ +1] = 0;
            Val * null_ptr = 0;
            if(!CAS(tmp_zoomed_vslot, &null_ptr, new_zoomed_vslot)){
                //already other thread allocate new slot
                free(new_zoomed_vslot);
            }
        }
        if(current_zoomed_level > zoomed_level){
            this->zoomed_level = current_zoomed_level;
        }
        //make sure that both Key and value slot exist
        K * k_slot = *tmp_zoomed_kslot;
        Val * v_slot = *tmp_zoomed_vslot;
        for(int j = 0; j < read_eplision_; j++){
            uint64_t empty = EMPTY_KEY;

            if(CAS((uint64_t *)&(k_slot[begin_addr+j]), &empty,key)){
                    counter_w[worker_id*3+2]++;
                    if(retraining_phase  && this->write_log != nullptr){
                        Val tmp_val = (val | LOG_MASK);
                        this->write_log->push_back(key, &v_slot[j+begin_addr]);
                        v_slot[j+begin_addr] = tmp_val;
                        return result_t::ok;
                    } 
                    //write number and buffer counter increase
                    v_slot[j+begin_addr] = val;
                    return result_t::ok;
                }
                if(k_slot[begin_addr+j] == key){
                    Val tmp_val = v_slot[j+begin_addr];
                    if(retraining_phase  && this->write_log != nullptr){
                        if(!(tmp_val & LOG_MASK)){
                            this->write_log->push_back(key, &v_slot[j+begin_addr]);
                        }
                        v_slot[j+begin_addr] = (val | LOG_MASK);
                        return result_t::ok;
                    } 
                    v_slot[j+begin_addr] = val;
                    return result_t::ok;
                }
            }
            //is full, try to find the next level
            tmp_zoomed_kslot = (K **)(&k_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
            tmp_zoomed_vslot = (Val **)(&v_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
        }
    return result_t::failed;
}

template<typename K, typename Val>
uint64_t Dnode<K, Val>::init_amount(){
    return init_size;
}

template<typename K, typename Val>
uint64_t Dnode<K, Val>::array_size(){
    return array_size_;
}

template<typename K, typename Val>
K Dnode<K, Val>::get_pivot(){
    return this->min_key;
}


template<typename K, typename Val> 
Dnode<K, Val> * Dnode<K, Val>::split_data_node(){
    //compact the data array and buffer
    std::vector<K> new_keys;
    std::vector<Val> val_ptr;

    size_t record_size = 0; 
    size_t count;
    new_keys.reserve(array_size_*2);
    val_ptr.reserve(array_size_*2);
    this->insertion_search_sort(new_keys,val_ptr, record_size);
    //add an test to identify the correctness of sort algorithm
    // if(new_keys[0] != this->min_key){
    //     DEBUG_THIS("[ERROR]: the sort is wrong ");
    //     abort();
    // }
    std::vector<segment> segments;
    std::vector<uint64_t> array_sizes;
    size_t read_eplision = this->read_eplision_;
    size_t train_eplision = this->epsilon_;
    double expansion_new = this->expansion_;
    if(this->read_intensive){
        read_eplision = 24;
        expansion_new = 1.5;
    }else if(this->write_intensive){
        read_eplision = 64;
        if(expansion_new < 3){
            expansion_new += 0.5;
        }
        
    }else{
        read_eplision = 32;
        expansion_new = 1.5;
    }
    //initilize the new data node
    size_t split_num = build_level(new_keys,train_eplision,segments, array_sizes);
    //to get the new data node
    Dnode * new_Dnode = new Dnode();
    Dnode * tmp_node = new_Dnode;
    new_Dnode->write_log = this->write_log;
    new_Dnode->training_node = this;
    new_Dnode->last = nullptr;
    new_Dnode->sync_phase = true;
    new_Dnode->new_retrained = true;
    size_t begin_addr = 0;
    Dnode * prev = nullptr;
    for(int i = 0; i < split_num; i++){
        if(begin_addr == new_keys.size()){
            prev->next = nullptr;
            break;
        }
        //only one node in the segment
        if(array_sizes[i] == 1){
           if(begin_addr != new_keys.size() - 1){
            DEBUG_THIS("only one key in here");
           }
            prev->next = nullptr;
            //add the only key into last node
            uint8_t log_c = this->worker_num - 1;
            prev->insert(new_keys[begin_addr], val_ptr[begin_addr], log_c, false);
            break;
        }
        if(array_sizes[i] == 0){
            DEBUG_THIS("the array size is 0");
            continue;
        }
        tmp_node->init(new_keys, val_ptr, expansion_new, train_eplision, segments[i], array_sizes[i], begin_addr, read_eplision, this->worker_num);
        tmp_node->last = prev;
        prev = tmp_node;
        if( i < split_num - 1){
            tmp_node->next = new Dnode();
            tmp_node = tmp_node->next;
        }
        begin_addr += array_sizes[i];
    }
    return new_Dnode;
}

template<typename K, typename Val> 
Dnode<K, Val> * Dnode<K, Val>::merge_data_node(Dnode * next_node){
    //if the next data node is retraining, skip
    if(next_node == nullptr){
        //just compact the single node
        return this->split_data_node();
    }else{
        //compress the node with the next node
        std::vector<K> new_keys_1;
        std::vector<Val> val_ptr_1;
        //read and sort the data in the two data node
        size_t record_size = 0; 
        size_t count;
        new_keys_1.reserve(array_size_*2);
        new_keys_1.reserve(array_size_*2);
        this->insertion_search_sort(new_keys_1,val_ptr_1, record_size);
        //add an test to identify the correctness of sort algorithm
        // if(new_keys_1[0] != this->min_key){
        //     DEBUG_THIS("[ERROR]: the sort is wrong ");
        //     abort();
        // }
        //
        std::vector<segment> segments;
        std::vector<uint64_t> array_sizes;
        //initilize the new data node
        size_t split_num = build_level(new_keys_1,epsilon_,segments, array_sizes);
        double expansion_new = this->expansion_;
        size_t read_eplision = this->read_eplision_;

        //to get the new data node
        Dnode * new_Dnode = new Dnode();
        Dnode * tmp_node = new_Dnode;
        new_Dnode->write_log = this->write_log;
        new_Dnode->training_node = this;
        new_Dnode->last = nullptr;
        new_Dnode->sync_phase = true;
         new_Dnode->new_retrained = true;
        size_t begin_addr = 0;
        Dnode * prev = nullptr;
        for(int i = 0; i < split_num; i++){
            if(begin_addr == new_keys_1.size()){
                prev->next = nullptr;
                break;
            }
            //only one node in the segment
            if(array_sizes[i] == 1){
            if(begin_addr != new_keys_1.size() - 1){
                DEBUG_THIS("only one key in here");
            }
                prev->next = nullptr;
                //add the only key into last node
                uint8_t log_c = this->worker_num - 1;
                prev->insert(new_keys_1[begin_addr], val_ptr_1[begin_addr], log_c);
                break;
            }
            if(array_sizes[i] == 0){
                DEBUG_THIS("the array size is 0");
                continue;
            }
            tmp_node->init(new_keys_1, val_ptr_1, expansion_new, epsilon_, segments[i], array_sizes[i], begin_addr, read_eplision, this->worker_num);
            tmp_node->last = prev;
            prev = tmp_node;
            if( i < split_num - 1){
                tmp_node->next = new Dnode();
                tmp_node = tmp_node->next;
            }
            begin_addr += array_sizes[i];
        }
        return new_Dnode;        
    }
}

/*
first do expansion, and try to insert the data in buffer to data array 
*/
template<typename K, typename Val> 
Dnode<K, Val> * Dnode<K, Val>::compact_phase(){
     //just first do the expansion
    //compact the data array and buffer
    std::vector<K> new_keys;
    std::vector<Val> val_ptr;

    size_t tmp_array_size = this->array_size_;
    size_t count;
    new_keys.reserve(array_size_ * 2);
    val_ptr.reserve(array_size_ * 2);
    //can only search the key, but not sort them 
    this->insertion_search_sort(new_keys,val_ptr, count);
    // if(new_keys[0] != this->min_key){
    //     DEBUG_THIS("[ERROR]: the sort is wrong ");
    //     abort();
    // }

    //initial with the old group
    Dnode<K,Val> * new_node = new Dnode(); 
    segment * tmp_seg = new segment(this->min_key, this->model->get_weight0(), this->model->get_weight1());
    new_node->min_key = this->min_key;
    new_node->array_size_ = this->array_size_;
    new_node->write_log = this->write_log;
    new_node->worker_num = this->worker_num;

    size_t read_eplision = this->read_eplision_;
    size_t expansion_new = this->expansion_;
    if(this->read_intensive){
        read_eplision = 24;
        expansion_new = 1.5;
    }else if(this->write_intensive){
        read_eplision = 64;
        if(expansion_new < 3 ){
            expansion_new += 0.5;
        }
   } else if(this->cold){
        read_eplision = 64;
        expansion_new = 0.6;
    }
    else{
        read_eplision = 32;
        expansion_new = 1.5;
    }
    //to determine wether this expansion is worthwhile
    new_node->expansion(new_keys, val_ptr, expansion_new, this->epsilon_, *tmp_seg, new_keys.size(), read_eplision);
    new_node->sync_phase = true;
    new_node->training_node = this;

    return new_node;
}

template<typename K, typename Val> 
Dnode<K, Val> * Dnode<K, Val>::new_compact_phase(){
    //just first do the expansion
    //compact the data array and buffer
    size_t tmp_array_size = this->array_size_;

    //initial with the old group
    Dnode<K,Val> * new_node = new Dnode(); 
    new_node->min_key = this->min_key;
    new_node->worker_num = this->worker_num;

    new_node->read_eplision_ = this->read_eplision_;
    new_node->expansion_ = this->expansion_;
    new_node->epsilon_ = this->epsilon_;
    if(this->read_intensive){
        read_eplision_ = 24;
        expansion_new = 2;
    }else if(this->write_intensive){
        read_eplision_ = 64;
        if(expansion_new < 4 ){
            expansion_new += 1;
        }
   } else if(this->cold){
        read_eplision_ = 48;
        expansion_new = 0.5;
    }
    else{
        read_eplision_ = 32;
        expansion_new = 1.5;
    }
    new_node->counter_r = new size_t[this->worker_num*3]();
    new_node->counter_w = new size_t[this->worker_num*3]();;

    size_t expansion_factor = new_node->expansion_;
    double tmp1 = this->model->get_weight0() * expansion_factor;
    int tmp2 = 0;
    new_node->model = new LinearRegressionModel<K>(tmp1, tmp2,this->min_key);
    new_node->array_size_ = (tmp_array_size + 2) * expansion_factor;
    size_t zoomed_amount = new_node->array_size_ / new_node->read_eplision_ + 1;
    new_node->zoomed_keys = new K*[zoomed_amount]();
    new_node->zoomed_vals = new Val*[zoomed_amount]();
    new_node->keys = new K[new_node->array_size_]();
    new_node->vals = new Val[new_node->array_size_];
    new_node->sync_phase = true;
    new_node->training_node = this;

    return new_node;

}

template<typename K, typename Val>
void Dnode<K,Val>::expansion(const std::vector<K> & key_begin,const std::vector<Val> & val_begin, double expansion,  size_t epsilon, segment seg, uint64_t array_size, size_t read_epsilon){
    //first get the useful information     
    size_t expansion_factor = expansion;

    this->read_eplision_ = read_epsilon;

    this->expansion_ = expansion_factor;

    this->epsilon_ = epsilon;
    this->init_size = array_size;
    //assume this is just 4, for one cache line
    
    this->counter_r = new size_t[this->worker_num*3]();
    this->counter_w = new size_t[this->worker_num*3]();

    double tmp1 = seg.slope * expansion_factor;
    int tmp2 = 0;
    K pivot_key = this->min_key;
    this->min_key = key_begin[0];
    this->array_size_ = (this->array_size_ + 2) * expansion_factor;
    this->model = new LinearRegressionModel<K>(tmp1, tmp2,seg.key);
    size_t zoomed_amount = this->array_size_ / ZOOM_SHARE + 1;
    this->zoomed_keys = new K*[zoomed_amount]();
    this->zoomed_vals = new Val*[zoomed_amount]();

    this->keys = new K[this->array_size_];
    this->vals = new Val[this->array_size_];
    //initial the buffer 
    double fill_ratio = (double)this->init_size / (double)this->array_size_;
    size_t buffer_size = 0;
    if(init_size == 0){
        DEBUG_THIS( " the init size is 0 in expansion" );
        abort();
    }
    int64_t relative_po = 0;
    int64_t pivot = -1;
    typename std::vector<K>::const_iterator iter_key = key_begin.begin();
    typename std::vector<Val>::const_iterator iter_val = val_begin.begin();
    //allocate the exact place & place the records, we need to go through
    for(int i = 0; i < array_size; i++){
        //convert to relative location
        relative_po = this->model->predict(*iter_key);
        //check if this is a collision
        //to search the free space in read_eplision 
        if(relative_po < array_size_ && relative_po > pivot){
            //just insert into the predicted posistion
            this->keys[relative_po] = *iter_key;
            this->vals[relative_po] = *iter_val;
            iter_key++;
            iter_val++;
        }else if(pivot < array_size_ && (pivot >= relative_po && (pivot - relative_po + 1) < read_eplision_)){
            relative_po = pivot +1;
            this->keys[relative_po] = *iter_key;
            this->vals[relative_po] = *iter_val;
            iter_key++;
            iter_val++;
        }else{
                buffer_size++;
                bool entered = false;
                 double full_lo = this->model->predict(*iter_key);
                //insert into buffer
                if(relative_po >= this->array_size_){
                    relative_po = this->array_size_;
                    full_lo == this->array_size_;
                }
                int loc = relative_po/ZOOM_SHARE;
                int begin_addr =  (relative_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)relative_po)*ZOOM_FACTOR;
                K** tmp_zoomed_kslot = &zoomed_keys[loc];
                Val** tmp_zoomed_vslot = &zoomed_vals[loc];
                int current_zoomed_level = 0;
                this->zoomed = true;
                while(1){
                    current_zoomed_level += 1;
                    if(*tmp_zoomed_kslot== 0){
                        //try to allocate the key region_slot, need to guarantee that there is only one
                        K * new_zoomed_kslot = new K[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 2]();
                        *tmp_zoomed_kslot = new_zoomed_kslot;
                    }
                    if(*tmp_zoomed_vslot == 0){
                        //other thread allocate the key region but not allocate the value region
                        Val * new_zoomed_vslot = new Val[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 2];
                        new_zoomed_vslot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 1] = 0;
                        *tmp_zoomed_vslot = new_zoomed_vslot;
                    }
                    if(current_zoomed_level > zoomed_level){
                        zoomed_level = current_zoomed_level;
                    }
                    //make sure that both Key and value slot exist
                    K * k_slot = *tmp_zoomed_kslot;
                    Val * v_slot = *tmp_zoomed_vslot;
                    for(int j = 0; j < read_eplision_; j++){
                        uint64_t empty = EMPTY_KEY;
                        if(k_slot[begin_addr+j] == empty){
                                k_slot[begin_addr+j] = *iter_key;
                                v_slot[j+begin_addr] = *iter_val;
                                entered = true;
                                break;
                            }

                        }
                        if(entered){
                            break;
                        }
                        //is full, try to find the next level
                        tmp_zoomed_kslot = (K **)(&k_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 1]);
                        tmp_zoomed_vslot = (Val **)(&v_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_+ 1]);
                    }
                iter_key++;
                iter_val++;
        }

        // this->keys[relative_po] = *iter_key;
        // this->vals[relative_po] = *iter_val;
        size_t array_loc = pivot+1;
        while(array_loc < relative_po){
            *((uint64_t*)&keys[array_loc]) = EMPTY_KEY;
            array_loc++;
        }    
        if(pivot < relative_po){
            pivot = relative_po;
        }
    }
    this->init_buffer_size = buffer_size;
    while(pivot < array_size_){
        *((uint64_t*)&keys[pivot]) = EMPTY_KEY;
        pivot++;
    }
}


template<typename K, typename Val>
void Dnode<K,Val>::free_data(){
    // delete [] this->keys;
    // delete [] this->vals;
}

template<typename K, typename Val>
void Dnode<K,Val>::free_log(){
    // this->write_log->free();
}
//needs to update the value also using CAS
template<typename K, typename Val>
bool Dnode<K, Val>::sync_log(const K &key, const Val &val){
    double full_lo = this->model->predict(key);
    size_t predict_po = full_lo;
    K min_key_in_model = this->min_key;
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
        full_lo = this->array_size_;
    }
    size_t insert_depth = 0;
    for(int k = 0; k < read_eplision_;k++){
        if(predict_po >= array_size_) break;
        uint64_t empty = EMPTY_KEY;
        if(keys[predict_po] == empty){
            if(CAS((uint64_t *)&keys[predict_po], &empty, key)){
                Val new_val = val | SHADOW_MASK;
                Val tmp_val = vals[predict_po];
                if(CAS((uint64_t *)&vals[predict_po], &tmp_val, new_val)){
                    return true;
                }else{
                    return false;
                }
                
            }
            if(empty == key){
                Val tmp_val = vals[predict_po];
                if(tmp_val & SHADOW_MASK){
                    return false;
                }
                Val new_val = val | SHADOW_MASK;
                if(CAS((uint64_t *)&vals[predict_po], &tmp_val, new_val)){
                    return true;
                }else{
                    return false;
                }
            }
        }
        //perform update
        if(keys[predict_po] == key){
            Val tmp_val = vals[predict_po];
            if(tmp_val & SHADOW_MASK){
                    return false;
                }
            Val new_val = val | SHADOW_MASK;
            if(CAS((uint64_t *)&vals[predict_po], &tmp_val, new_val)){
                return true;
            }else{
                return false;
            }
        }
        predict_po ++;
        insert_depth++;
    }
    //insert to the zoomed area
    if(!zoomed){
        zoomed = true;
    }
    //whether this slot is created
    predict_po = full_lo;
    int loc = predict_po/ZOOM_SHARE;
    int begin_addr = (predict_po % ZOOM_SHARE)*ZOOM_FACTOR + (full_lo - (double)predict_po)*ZOOM_FACTOR;

    K** tmp_zoomed_kslot = &zoomed_keys[loc];
    Val** tmp_zoomed_vslot = &zoomed_vals[loc];
    int current_zoomed_level = 0;
    
    while(1){
        current_zoomed_level += 1;
        if( *tmp_zoomed_kslot == 0){
            //try to allocate the key region_slot, need to guarantee that there is only one
            K * new_zoomed_kslot = new K[ZOOM_SHARE * ZOOM_FACTOR +read_eplision_ +2]();
            K * null_ptr = 0;
            if(!CAS(tmp_zoomed_kslot, &null_ptr, new_zoomed_kslot)){
                delete[] new_zoomed_kslot;
            }
        }
        if(*tmp_zoomed_vslot == 0){
            //other thread allocate the key region but not allocate the value region
            Val * new_zoomed_vslot = new Val[ZOOM_SHARE * ZOOM_FACTOR +read_eplision_ +2];
            new_zoomed_vslot[ZOOM_SHARE * ZOOM_FACTOR +read_eplision_ +1] = 0;
            Val * null_ptr = nullptr;
            if(!CAS(tmp_zoomed_vslot, &null_ptr, new_zoomed_vslot)){
                //already other thread allocate new slot
                delete[] new_zoomed_vslot;
            }
        }
        if(current_zoomed_level > zoomed_level){
            this->zoomed_level = current_zoomed_level;
        }
        //make sure that both Key and value slot exist
        K * k_slot = *tmp_zoomed_kslot;
        Val * v_slot = *tmp_zoomed_vslot;

        for(int j = 0; j < read_eplision_; j++){
            uint64_t empty = EMPTY_KEY;
            if(CAS((uint64_t *)&(k_slot[begin_addr+j]), &empty,key)){
                    //write number and buffer counter increase
                    Val new_val = val | SHADOW_MASK;
                    Val tmp_val = v_slot[j+begin_addr];
                    if(CAS((uint64_t *)&v_slot[j+begin_addr], &tmp_val, new_val)){
                        return true;
                    }else{
                        return false;
                    }
                }
                if(k_slot[begin_addr+j] == key){
                    Val tmp_val = v_slot[j+begin_addr];
                    if(tmp_val & SHADOW_MASK){
                        return false;
                    }
                    Val new_val = val | SHADOW_MASK;
                    if(CAS((uint64_t *)&v_slot[j+begin_addr], &tmp_val, new_val)){
                        return true;
                    }else{
                        return false;
                    }
                }
            }
            //is full, try to find the next level
            tmp_zoomed_kslot = (K **)(&k_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
            tmp_zoomed_vslot = (Val **)(&v_slot[(ZOOM_SHARE) * ZOOM_FACTOR + read_eplision_ + 1]);
        }
    return false;
}
template<typename K, typename Val>
void Dnode<K, Val>::sync_reocrd(){
    K tmp_key = 0;
    Val tmp_val = 0;
    K* key_bucket = nullptr;
    Val * val_bucket = nullptr;
    Dnode * trained_node = this->training_node;
    size_t buffer_size = 0;
    size_t counter_size = 0;
    size_t old_read_region = trained_node->read_eplision_;
    size_t new_read_region = this->read_eplision_;
    uint64_t node_size = trained_node->array_size_;
    for(int i = 0; i < node_size; i++){
        tmp_key = trained_node->keys[i];
        tmp_val = trained_node->vals[i];
        if(tmp_key != EMPTY_KEY){
            bool enter_bucket = false;
            if(insert_record(tmp_key,tmp_val,enter_bucket)){
                counter_size++;
                if(enter_bucket){
                    buffer_size++;
                }
            }
        }
        //insert the records in the zoomed buckets
        if(i%old_read_region == 0){
            size_t bucket_num = i/old_read_region;
            key_bucket = trained_node->zoomed_keys[bucket_num];
            val_bucket = trained_node->zoomed_vals[bucket_num];            
        
            if(key_bucket && val_bucket){
                while(1){
                    for(int k = 0; k < ZOOM_FACTOR * old_read_region; k++){
                        tmp_key = key_bucket[k];
                        tmp_val = val_bucket[k];
                        if(tmp_key != EMPTY_KEY){
                            bool enter_bucket = false;
                            if(insert_record(tmp_key,tmp_val,enter_bucket)){
                                counter_size++;
                                if(enter_bucket){
                                    buffer_size++;
                                }
                            }
                        }
                    }
                    if(key_bucket[ZOOM_FACTOR*old_read_region] != 0 && val_bucket[ZOOM_FACTOR*old_read_region] != 0){
                        key_bucket = (K *)(key_bucket[ZOOM_FACTOR*old_read_region]);
                        val_bucket = (Val *)(val_bucket[ZOOM_FACTOR*old_read_region]);
                    }else{
                        break;
                    }
                }
            }  
        }  
        // this->sync_pivot++;
    }

    //get the init information
    this->init_buffer_size = buffer_size;
    this->init_size = counter_size;    
}
template<typename K, typename Val>
bool Dnode<K, Val>::insert_record(const K &key, const Val &val, bool & bucket){
    size_t predict_po = this->model->predict(key);

    for(int k = 0; k < this->read_eplision_; k++){
        if(predict_po >= this->array_size_) break;
        uint64_t empty = EMPTY_KEY;
        if(CAS((uint64_t *)&keys[predict_po], &empty, key)){
            //already insret the records into the new data node
            vals[predict_po] = val;
            return true;
        }
        //continue to identify whether the record is already existen
        if(empty == key){
            return false;
        }
        predict_po ++;
    }
    //insert into the bucket
    if(!this->zoomed){
        this->zoomed = true;
    }
    predict_po = this->model->predict(key);
    if(predict_po >= this->array_size_){
        predict_po = this->array_size_;
    }
    int loc = predict_po/this->read_eplision_;
    if(this->insert_to_bucket(key, val, loc, predict_po)){
        bucket = true;
        return true;
    }
    return false;
}
template<typename K, typename Val>
bool Dnode<K, Val>::insert_to_bucket(const K &key,const Val &val, size_t loc, size_t predict_po){
    int begin_addr = (predict_po % read_eplision_)*ZOOM_FACTOR;
    K** tmp_zoomed_kslot = &zoomed_keys[loc];
    Val** tmp_zoomed_vslot = &zoomed_vals[loc];
    int current_zoomed_level = 0;
    while(1){
        if(*tmp_zoomed_kslot == nullptr || *tmp_zoomed_kslot == 0){
            K * new_zoomed_kslot = new K[ZOOM_FACTOR * read_eplision_ + 1]();
            //need to initilize the slot
            // for(int i = 0; i < ZOOM_FACTOR * read_eplision_; i++){
            //     *((uint64_t*)&new_zoomed_kslot[i]) = EMPTY_KEY;
            // }
            // new_zoomed_kslot[ZOOM_FACTOR * read_eplision_] = 0;
            K * null_ptr = 0;
            if(!CAS(tmp_zoomed_kslot, &null_ptr, new_zoomed_kslot)){
            }
        }
        current_zoomed_level += 1;
        if(current_zoomed_level > zoomed_level){
            this->zoomed_level = current_zoomed_level;
        }
        if(*tmp_zoomed_vslot == nullptr || *tmp_zoomed_vslot == 0){
            //other thread allocate the key region but not allocate the value region
            Val * new_zoomed_vslot = new Val[ZOOM_FACTOR * read_eplision_ + 1];
            new_zoomed_vslot[ZOOM_FACTOR * read_eplision_] = 0;
            Val * null_ptr = nullptr;
            if(!CAS(tmp_zoomed_vslot, &null_ptr, new_zoomed_vslot)){
            }
        }
        K * k_slot = *tmp_zoomed_kslot;
        Val * v_slot = *tmp_zoomed_vslot;
        for(int j = 0; j < ZOOM_FACTOR; j++){
            uint64_t empty = EMPTY_KEY;
            if(CAS((uint64_t *)&(k_slot[begin_addr+j]), &empty,key)){
                v_slot[j+begin_addr] = val;
                return true;
            }
            if(k_slot[begin_addr+j] == key){
                return false;
            }
        }
        tmp_zoomed_kslot = (K **)(&k_slot[ZOOM_FACTOR * read_eplision_]);
        tmp_zoomed_vslot = (Val **)(&v_slot[ZOOM_FACTOR * read_eplision_]);
    } 
    return false;   
}

template<typename K, typename Val>
void Dnode<K, Val>::insertion_search_sort(std::vector<K> & new_keys,std::vector<Val> & new_vals, size_t & n){
    size_t loc_new  = 0;
    size_t key_in_array = 0;
    size_t zoomed_number =this->array_size_/ZOOM_SHARE + 1;
    size_t largest_level = this->zoomed_level;
    for(size_t zoomed_counter = 0; zoomed_counter < zoomed_number; zoomed_counter ++){
        //sort the data in zoomed slot
        //snapshot the zoomed level
        if(zoomed_keys[zoomed_counter]!=nullptr && zoomed_keys[zoomed_counter]!=0){
            size_t level = 0;
            K * tmp_zoom_K = zoomed_keys[zoomed_counter];
            K * tmp_zoom_V = zoomed_vals[zoomed_counter];
            //take a snapshot
            K * next_zoom_K;
            K * next_zoom_V;
            level = 0;
            while(tmp_zoom_K != nullptr && tmp_zoom_V != nullptr ){
                level++;
                if(level > largest_level){
                    break;
                }
                next_zoom_K = (K *) * (&tmp_zoom_K[(ZOOM_FACTOR) * ZOOM_SHARE + read_eplision_ + 1]);
                next_zoom_V = (Val *) * (&tmp_zoom_V[(ZOOM_FACTOR) * ZOOM_SHARE + read_eplision_ + 1]);
                for(int j = 0; j < (ZOOM_FACTOR) * ZOOM_SHARE + read_eplision_ ; j++){
                    if(*((uint64_t *)&tmp_zoom_K[j]) != EMPTY_KEY ){
                        K ret = tmp_zoom_K[j];
                        new_keys.push_back(0);
                        new_vals.push_back(0);
                        int64_t index = loc_new - 1;
                        while(index >= 0 && new_keys[index] > ret){
                            new_keys[index+1] = new_keys[index];
                            new_vals[index+1] = new_vals[index];
                            index--;
                        }
                        new_keys[index+1] = ret;
                        //val equal to 1
                        new_vals[index+1] = (tmp_zoom_K[j] & 0x3FFFFFFFFFFFFFFF);
                        loc_new++;
                    }
                }
                tmp_zoom_K = next_zoom_K;
                tmp_zoom_V = next_zoom_V;
                
            }
        }
        //compact the data in the array
        for(int i = 0; i < ZOOM_SHARE; i++){
            if(key_in_array == this->array_size_){
                break;
            }
            if(*((uint64_t *)&keys[key_in_array]) != EMPTY_KEY){
                K ret = keys[key_in_array];
                new_keys.push_back(0);
                new_vals.push_back(0);

                int64_t index = loc_new - 1;
                while(index >= 0 && new_keys[index] > ret){
                    new_keys[index+1] = new_keys[index];
                    new_vals[index+1] = new_vals[index];
                    index--;
                }
                new_keys[index+1] = ret;
                new_vals[index+1] = (vals[key_in_array] & 0x3FFFFFFFFFFFFFFF);
                loc_new++;
            }
            key_in_array++;
        }
    }
    n = loc_new;
}

template<typename K, typename Val>
void Dnode<K, Val>::insertion_search(std::vector<K> & new_keys,std::vector<Val> & new_vals, size_t & n){
    size_t loc_new  = 0;
    size_t key_in_array = 0;
    size_t zoomed_number = this->array_size_/this->read_eplision_ + 1;
    size_t largest_level = this->zoomed_level;
    for(size_t zoomed_counter = 0; zoomed_counter < zoomed_number; zoomed_counter ++){
        //sort the data in zoomed slot
        //snapshot the zoomed level
        if(zoomed_keys[zoomed_counter]!=nullptr && zoomed_keys[zoomed_counter]!= 0){
            size_t level = 0;
            K * tmp_zoom_K = zoomed_keys[zoomed_counter];
            K * tmp_zoom_V = zoomed_vals[zoomed_counter];
            //take a snapshot
            K * next_zoom_K;
            K * next_zoom_V;
            level = 0;
            while(tmp_zoom_K != nullptr && tmp_zoom_V != nullptr && level <= largest_level){
                next_zoom_K = (K *) * (&tmp_zoom_K[this->epsilon_ * ZOOM_FACTOR]);
                next_zoom_V = (Val *) * (&tmp_zoom_V[this->epsilon_ * ZOOM_FACTOR]);
                for(int j = 0; j < this->epsilon_ * ZOOM_FACTOR ; j++){
                    if(*((uint64_t *)&tmp_zoom_K[j]) != EMPTY_KEY ){
                        K ret = tmp_zoom_K[j];
                        new_keys.push_back(ret);
                        new_vals.push_back(tmp_zoom_K[j] & 0x7FFFFFFFFFFFFFFF);
                        loc_new++;
                    }
                }
                tmp_zoom_K = next_zoom_K;
                tmp_zoom_V = next_zoom_V;
                level++;
            }
        }
        //compact the data in the array
        for(int i = 0; i < this->read_eplision_; i++){
            if(key_in_array == this->array_size_){
                break;
            }
            if(*((uint64_t *)&keys[key_in_array]) != EMPTY_KEY){
                K ret = keys[key_in_array];
                new_keys.push_back(ret);
                new_vals.push_back(vals[key_in_array] & 0x7FFFFFFFFFFFFFFF);
            }
            key_in_array++;
            
        }
    }
    n = loc_new;
}

}