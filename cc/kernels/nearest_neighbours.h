#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {
  typedef struct MetalPlugin {} MetalDevice;
}

namespace tensorflow {

  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;
  typedef Eigen::MetalDevice MetalDevice;

  namespace functor {

    template<typename Device, typename T>
    struct NearestNeighboursFunctor {
      void operator()(
          const Device &d,
          const int32_t batch_size,
          const int32_t num_tokens,
          const int32_t vocab_size,
          const int32_t embedding_dim,
          const T *token_embeddings,
          const T *embedding_matrix,
          T *output
      );
    };
  }
}

