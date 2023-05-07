#pragma once

#include <concepts>
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


namespace tensorflow::functor {

  template<const int N, typename Device, typename T> requires std::floating_point<T>
  struct NearestNeighboursIndexesFunctor {
    void operator()(
        const Device &d,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        int *output
    );
  };

  template<const int N, typename Device, typename T> requires std::floating_point<T>
  struct NearestNeighboursFunctor {
    void operator()(
        const Device &d,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        T *output
    );
  };


}

