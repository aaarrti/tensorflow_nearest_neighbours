#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "nearest_neighbours.h"


#ifdef CUDA
#define EIGEN_USE_GPU


namespace tensorflow {
  namespace functor {

    typedef Eigen::GpuDevice GPUDevice;

    // Define the CUDA kernel.
    template<typename T>
    __global__ void NearestNeighboursCudaKernel(
        //const int32_t vocab_size,
        //const int32_t embedding_dim,
        //const tensorflow::Tensor *token_embeddings,
        //const tensorflow::Tensor *embedding_matrix_shaped,
        //const Eigen::Matrix<const T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_embedding_matrix,
        //tensorflow::Tensor *output_shaped
    ) {

      //const auto sequence = Eigen::Map<
      //    const Eigen::Matrix <const T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      //>(token_embeddings->SubSlice(threadIdx.x).flat<T>().data(), vocab_size, embedding_dim);
      //const auto distances = std::vector<T>(vocab_size);
      //const auto embedding = sequence.row(threadIdx.y);
      // Find index of the smallest distance
      //const auto argmin = nearest_neighbour_index<T>(vocab_size, embedding, eigen_embedding_matrix);
      // Fill the output
      //for (auto i = 0; i != embedding_dim; i++) {
      //  output_shaped({threadIdx.x, threadIdx.y, i}) = embedding_matrix_shaped({argmin, i});
      //}
    }

    template<typename T>
    struct NearestNeighboursFunctor<GPUDevice, T> {

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

        auto embedding_matrix_shaped = embedding_matrix->shaped<T, 2>({vocab_size, embedding_dim});
        const auto eigen_embedding_matrix = Eigen::Map<
            const Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        >(embedding_matrix->flat<T>().data(), vocab_size, embedding_dim);

        auto output_shaped = output_tensor->shaped<T, 3>({batch_size, sequence_length, embedding_dim});

        //NearestNeighboursCudaKernel<float><<<block_count, thread_per_block, 0, device.stream()>>>(
        //    vocab_size, embedding_dim, token_embeddings, embedding_matrix_shaped, eigen_embedding_matrix, output_shaped);
      }
    };

    // Explicitly instantiate functors for the types of OpKernels registered.
    template struct NearestNeighboursFunctor<GPUDevice, float>;
  }
}

#endif