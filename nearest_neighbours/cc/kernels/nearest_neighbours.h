#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename T>
struct NearestNeighboursFunctor {
  void operator()(const Device &d, int batch_size, int num_tokens,
                  int vocab_size, int embedding_dim, const T *token_embeddings,
                  const T *embedding_matrix, T *output);
};
}  // namespace functor
}  // namespace tensorflow



