#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "Eigen/CXX11/Tensor"

#ifdef CUDA
#define EIGEN_USE_GPU
#endif


namespace tensorflow {
  namespace functor {

    typedef Eigen::GpuDevice GPUDevice;

    // Define the CUDA kernel.
    __global__ void NearestNeighboursCudaKernel() {}


    template<>
    struct NearestNeighboursFunctor<GPUDevice> {
      void operator()(
          const GPUDevice &device,
          const tensorflow::Tensor *token_embeddings,
          const tensorflow::Tensor *embedding_matrix,
          tensorflow::Tensor *output_tensor
      ) {

        const auto batch_size = static_cast<int32_t>(token_embeddings->dim_size(0));
        const auto vocab_size = static_cast<int32_t>(embedding_matrix->dim_size(0));
        const auto sequence_length = static_cast<int32_t>(token_embeddings->dim_size(1));
        const auto embedding_dim = static_cast<int32_t>(token_embeddings->dim_size(2));

        const auto block_count = batch_size;
        const auto thread_per_block = sequence_length;

        auto embedding_matrix_shaped = embedding_matrix->shaped<float, 2>({vocab_size, embedding_dim});

        NearestNeighboursCudaKernel<<<block_count, thread_per_block, 0, device.stream()>>>();
      }
    };

// Explicitly instantiate functors for the types of OpKernels registered.
    template<>
    struct NearestNeighboursFunctor<GPUDevice>;
  }
}