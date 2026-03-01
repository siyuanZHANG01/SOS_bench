#include "model.h"
#pragma once
namespace loft{
    
template <typename K>
LinearRegressionModel<K>::LinearRegressionModel(double w, int i, K key) {
    min_key = key;
    a = w;
    b = i;
}

template <typename K>
LinearRegressionModel<K>::LinearRegressionModel() {}

template <typename K>
LinearRegressionModel<K>::~LinearRegressionModel(){}

template <typename K>
double LinearRegressionModel<K>::predict(const K key){
    double pos = ((a * (key-min_key)) + b);
    return pos > 0 ? (pos) : 0;
}

template <typename K>
size_t LinearRegressionModel<K>::predict_rmi(const K key){
  model_key_t model_key;
  model_key[0] = key;
  double *model_key_ptr = model_key.data();
  double res = weights[0] * *model_key_ptr + weights[1];
  return res > 0 ? res : 0;
}

template <class K>
void LinearRegressionModel<K>::prepare(const std::vector<K> &keys,
                                const std::vector<size_t> &positions) {
  assert(keys.size() == positions.size());
  if (keys.size() == 0) return;

  std::vector<model_key_t> model_keys(keys.size());
  std::vector<double *> key_ptrs(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    model_key_t model_key;
    model_key[0] = keys[i];
    model_keys[i] = model_key;
    key_ptrs[i] = model_keys[i].data();
  }
  this->min_key = keys[0];
  prepare_model(key_ptrs, positions);
}
template <class K>
void LinearRegressionModel<K>::prepare_model(
    const std::vector<double *> &model_key_ptrs,
    const std::vector<size_t> &positions) {
  size_t key_len = 1;
  if (positions.size() == 0) return;
  if (positions.size() == 1) {
    LinearRegressionModel<K>::weights[key_len] = positions[0];
    return;
  }

  if (key_len == 1) {  // use multiple dimension LR when running tpc-c
    double x_expected = 0, y_expected = 0, xy_expected = 0,
           x_square_expected = 0;
    for (size_t key_i = 0; key_i < positions.size(); key_i++) {
      double key = model_key_ptrs[key_i][0];
      x_expected += key;
      y_expected += positions[key_i];
      x_square_expected += key * key;
      xy_expected += key * positions[key_i];
    }
    x_expected /= positions.size();
    y_expected /= positions.size();
    x_square_expected /= positions.size();
    xy_expected /= positions.size();

    weights[0] = (xy_expected - x_expected * y_expected) /
                 (x_square_expected - x_expected * x_expected);
    weights[1] = (x_square_expected * y_expected - x_expected * xy_expected) /
                 (x_square_expected - x_expected * x_expected);
    return;
  }

  // trim down samples to avoid large memory usage
  size_t step = 1;
  if (model_key_ptrs.size() > desired_training_key_n) {
    step = model_key_ptrs.size() / desired_training_key_n;
  }

  std::vector<size_t> useful_feat_index;
  for (size_t feat_i = 0; feat_i < key_len; feat_i++) {
    double first_val = model_key_ptrs[0][feat_i];
    for (size_t key_i = 0; key_i < model_key_ptrs.size(); key_i += step) {
      if (model_key_ptrs[key_i][feat_i] != first_val) {
        useful_feat_index.push_back(feat_i);
        break;
      }
    }
  }
  if (model_key_ptrs.size() != 1 && useful_feat_index.size() == 0) {
    COUT_THIS("all feats are the same");
  }
  size_t useful_feat_n = useful_feat_index.size();
  bool use_bias = true;

  // we may need multiple runs to avoid "not full rank" error
  int fitting_res = -1;
  while (fitting_res != 0) {
    // use LAPACK to solve least square problem, i.e., to minimize ||b-Ax||_2
    // where b is the actual positions, A is inputmodel_keys
    int m = model_key_ptrs.size() / step;                  // number of samples
    int n = use_bias ? useful_feat_n + 1 : useful_feat_n;  // number of features
    double *a = (double *)malloc(m * n * sizeof(double));
    double *b = (double *)malloc(std::max(m, n) * sizeof(double));
    if (a == nullptr || b == nullptr) {
      COUT_N_EXIT("cannot allocate memory for matrix a or b");
    }

    for (int sample_i = 0; sample_i < m; ++sample_i) {
      // we only fit with useful features
      for (size_t useful_feat_i = 0; useful_feat_i < useful_feat_n;
           useful_feat_i++) {
        a[sample_i * n + useful_feat_i] =
            model_key_ptrs[sample_i * step][useful_feat_index[useful_feat_i]];
      }
      if (use_bias) {
        a[sample_i * n + useful_feat_n] = 1;  // the extra 1
      }
      b[sample_i] = positions[sample_i * step];
      assert(sample_i * step < model_key_ptrs.size());
    }

    // fill the rest of b when m < n, otherwise nan value will cause failure
    for (int b_i = m; b_i < n; b_i++) {
      b[b_i] = 0;
    }

    fitting_res = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, 1 /* nrhs */, a,
                                n /* lda */, b, 1 /* ldb, i.e. nrhs */);

    if (fitting_res > 0) {
      // now we need to remove one column in matrix a
      // note that fitting_res indexes starting with 1
      if ((size_t)fitting_res > useful_feat_index.size()) {
        use_bias = false;
      } else {
        size_t feat_i = fitting_res - 1;
        useful_feat_index.erase(useful_feat_index.begin() + feat_i);
        useful_feat_n = useful_feat_index.size();
      }

      if (useful_feat_index.size() == 0 && use_bias == false) {
        COUT_N_EXIT(
            "impossible! cannot fail when there is only 1 bias column in "
            "matrix a");
      }
    } else if (fitting_res < 0) {
      printf("%i-th parameter had an illegal value\n", -fitting_res);
      exit(-2);
    }

    // set weights to all zero
    for (size_t weight_i = 0; weight_i < weights.size(); weight_i++) {
      weights[weight_i] = 0;
    }
    // set weights of useful features
    for (size_t useful_feat_i = 0; useful_feat_i < useful_feat_index.size();
         useful_feat_i++) {
      weights[useful_feat_index[useful_feat_i]] = b[useful_feat_i];
    }
    // set bias
    if (use_bias) {
      size_t key_len = 1;
      weights[key_len] = b[n - 1];
    }

    free(a);
    free(b);
  }
  assert(fitting_res == 0);
}

// template <typename K>
// bool LinearRegressionModel<K>::position(const K key){
//     return ((expansion * a) * (key - min_key) +b) <= ((size_t)(expansion * init_size) >>1) ? true : false;
//}

template <typename K>
std::vector<size_t> LinearRegressionModel<K>::predict(const std::vector<K> &keys){
    int num = keys.size();
    std::vector<size_t> predict_po(num);
    for(int i = 0; i < num; i++){
        predict_po.push_back(size_t(this->a * keys[i] + this->b));
    }
    return predict_po;
}

}