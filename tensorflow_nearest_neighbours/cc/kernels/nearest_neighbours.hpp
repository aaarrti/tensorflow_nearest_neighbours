#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


namespace tensorflow {

  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;

  namespace functor {

    struct paramsType {
      const int batch_size;
      const int num_tokens;
      const int vocab_size;
      const int embedding_dim;
    };

    typedef struct paramsType paramsType;

    template<typename Device, typename T>
    struct NearestNeighboursFunctor {
      void
      operator()(const Device &d, const paramsType paramsType, const T *token_embeddings,
                 const T *embedding_matrix, T *output);
    };
  }
}

