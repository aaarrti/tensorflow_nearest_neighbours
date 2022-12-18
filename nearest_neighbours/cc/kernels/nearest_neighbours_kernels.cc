#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "nearest_neighbours/cc/kernels/nearest_neighbours.h"

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif


namespace tensorflow {

  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;

  namespace functor {

    template<>
    struct NearestNeighboursFunctor<CPUDevice> {
      void operator()(const CPUDevice &device,
                      const tensorflow::Tensor *token_embeddings,
                      const tensorflow::Tensor *embedding_matrix,
                      tensorflow::Tensor *output_tensor) {
        // Get input dims
        const auto batch_size = static_cast<int32_t>(token_embeddings->dim_size(0));
        const auto vocab_size = static_cast<int32_t>(embedding_matrix->dim_size(0));
        const auto sequence_length = static_cast<int32_t>(token_embeddings->dim_size(1));
        const auto embedding_dim = static_cast<int32_t>(token_embeddings->dim_size(2));
        // Shape Input
        auto embedding_matrix_shaped = embedding_matrix->shaped<float, 2>({vocab_size, embedding_dim});

        // Reshape for indexing
        const auto output_shaped = output_tensor->shaped<float, 3>(
            {batch_size, sequence_length, embedding_dim}
        );

        // Convert to Eigen::Matrix for computation
        const auto eigen_embedding_matrix = Eigen::Map<
            const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        >(embedding_matrix->flat<float>().data(), vocab_size, embedding_dim);

        for (auto batch_index = 0; batch_index != batch_size; batch_index++) {
          const auto sequence = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
              token_embeddings->SubSlice(batch_index).flat<float>().data(), vocab_size, embedding_dim
          );

          for (auto token_index = 0; token_index != sequence_length; token_index++) {
            auto distances = std::vector<float>(vocab_size);

            const auto embedding = sequence.row(token_index);

            // Find index of the smallest distance
            auto argmin = nearest_neighbour_index(vocab_size, embedding, eigen_embedding_matrix);

            // Fill the output
            for (auto i = 0; i != embedding_dim; i++) {
              output_shaped({batch_index, token_index, i}) = embedding_matrix_shaped({argmin, i});
            }
          }
        }
      }
    };


    template<typename Device>
    class NearestNeighboursOp : public tensorflow::OpKernel {
    public:
      explicit NearestNeighboursOp(tensorflow::OpKernelConstruction *context) : tensorflow::OpKernel(context) {}

      void Compute(tensorflow::OpKernelContext *context) override {
        // Create inputs
        const tensorflow::Tensor *token_embeddings = nullptr;
        const tensorflow::Tensor *embedding_matrix = nullptr;
        // Check inputs were passed
        OP_REQUIRES_OK(context, context->input("token_embeddings", &token_embeddings));
        OP_REQUIRES_OK(context, context->input("embedding_matrix", &embedding_matrix));

        // Create output
        tensorflow::Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, token_embeddings->shape(), &output_tensor));

        NearestNeighboursFunctor<Device>()(
            context->eigen_device<Device>(),
            token_embeddings,
            embedding_matrix,
            output_tensor
        );
      }
    };


    // Register the CPU kernels.
#define REGISTER_CPU() REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device(DEVICE_CPU), NearestNeighboursOp<CPUDevice>);
    REGISTER_CPU()


#ifdef GOOGLE_CUDA
#define REGISTER_GPU()  extern template struct NearestNeighboursFunctor<GPUDevice>; \
                          REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device(DEVICE_GPU), NearestNeighboursOp<GPUDevice>);


    REGISTER_GPU()
#endif


  }
}