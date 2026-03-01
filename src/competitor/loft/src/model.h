// #ifndef __MODEL_H__
// #define __MODEL_H__

// #include "mkl.h"
// #include "mkl_lapacke.h"
#include <array>
#include "mkl.h"
#include "mkl_lapacke.h"
#include <vector>
#include <numeric>
#pragma once
namespace loft{
template <typename K>
class LinearRegressionModel{
    static const size_t desired_training_key_n = 10000000;
    typedef std::array<double, 1> model_key_t;
public:
    inline LinearRegressionModel();
    inline LinearRegressionModel(double w, int i, K key);
    ~LinearRegressionModel();
    void prepare(const std::vector<K> &keys,
               const std::vector<size_t> &positions);
    void prepare_model(const std::vector<double *> &model_key_ptrs, const std::vector<size_t> &positions);
    double predict(const K key);
    size_t predict_rmi(const K keys);
    bool position(const K key);
    std::vector<size_t> predict(const std::vector<K> &keys);
    void set_begin_addr(size_t x){begin_addr = x;}
    inline size_t get_begin_addr(void){ return begin_addr;}
    inline double get_weight0(){ return a; }
    inline int get_weight1(){ return b; }
    inline K get_pivot(){return min_key;}
    inline void set_expansion(double k){expansion = k;}
    size_t begin_addr = 0;
private:
    K min_key;
    double a;
    int b;
    double expansion = 1.0;
    std::array<double, 2> weights;
};
}