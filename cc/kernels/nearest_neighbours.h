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
          const tensorflow::Tensor *token_embeddings,
          const tensorflow::Tensor *embedding_matrix,
          tensorflow::Tensor *output_tensor
      );
    };
  }
}

