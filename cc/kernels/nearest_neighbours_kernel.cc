#include "nearest_neighbours.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"

namespace tensorflow {

namespace functor {

template <typename T>
int32_t nearest_neighbour_index(
    const int32_t vocab_size, const Eigen::Vector<T, Eigen::Dynamic> &embedding,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        &embedding_matrix) {
  auto distances = std::vector<T>(vocab_size);

  const auto embedding_row_major = embedding.transpose();

  for (auto matrix_row_index = 0; matrix_row_index != vocab_size;
       matrix_row_index++) {
    // Compute distance between current embedding and each matrix row
    const auto row = embedding_matrix.row(matrix_row_index);
    const auto dist = static_cast<T>((row - embedding_row_major).squaredNorm());
    distances[matrix_row_index] = dist;
  }

  // Find index of the smallest distance
  const auto it = std::min_element(std::begin(distances), std::end(distances));
  const auto argmin =
      static_cast<int32_t>(std::distance(std::begin(distances), it));
  return argmin;
}

int32_t index_2d_flat(int32_t index_0, int32_t index_1, int32_t shape_1) {
  return index_1 + index_0 * shape_1;
}

int32_t index_3d_flat(int32_t index_0, int32_t index_1, int32_t index_2,
                      int32_t shape_1, int32_t shape_2) {
  return index_2 + index_1 * shape_2 + index_0 * shape_2 * shape_1;
}

template <typename T> struct NearestNeighboursFunctor<CPUDevice, T> {
  void operator()(const CPUDevice &device, const int32_t batch_size,
                  const int32_t num_tokens, const int32_t vocab_size,
                  const int32_t embedding_dim, const T *token_embeddings,
                  const T *embedding_matrix, T *output) {

    // Convert to Eigen::Matrix for computation
    const auto eigen_embedding_matrix =
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(
            embedding_matrix, vocab_size, embedding_dim);

    for (auto batch_index = 0; batch_index != batch_size; batch_index++) {
      const auto sequence =
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>(
              token_embeddings +
                  index_3d_flat(batch_index, 0, 0, num_tokens, embedding_dim),
              vocab_size, embedding_dim);

      for (auto token_index = 0; token_index != num_tokens; token_index++) {
        auto distances = std::vector<T>(vocab_size);

        const auto embedding = sequence.row(token_index);

        // Find index of the smallest distance
        auto argmin = nearest_neighbour_index<T>(vocab_size, embedding,
                                                 eigen_embedding_matrix);

        // Fill the output
        for (auto i = 0; i != embedding_dim; i++) {
          output[index_3d_flat(batch_index, token_index, i, num_tokens,
                               embedding_dim)] =
              embedding_matrix[index_2d_flat(argmin, i, embedding_dim)];
        }
      }
    }
  }
};

template <typename Device, typename T>
class NearestNeighboursOp : public OpKernel {
public:
  explicit NearestNeighboursOp(OpKernelConstruction *context): OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Create inputs
    const Tensor *token_embeddings = nullptr;
    const Tensor *embedding_matrix = nullptr;
    // Check inputs were passed
    OP_REQUIRES_OK(context,
                   context->input("token_embeddings", &token_embeddings));
    OP_REQUIRES_OK(context,
                   context->input("embedding_matrix", &embedding_matrix));

    // Create output
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, token_embeddings->shape(), &output_tensor));

    const auto batch_size = static_cast<int32_t>(token_embeddings->dim_size(0));
    const auto vocab_size = static_cast<int32_t>(embedding_matrix->dim_size(0));
    const auto num_tokens = static_cast<int32_t>(token_embeddings->dim_size(1));
    const auto embedding_dim =
        static_cast<int32_t>(token_embeddings->dim_size(2));

    NearestNeighboursFunctor<Device, T>()(
        context->eigen_device<Device>(), batch_size, num_tokens, vocab_size,
        embedding_dim, token_embeddings->flat<T>().data(),
        embedding_matrix->flat<T>().data(), output_tensor->flat<T>().data());
  }
};

#define REGISTER_CPU()                                                         \
  REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device("CPU"),             \
                          NearestNeighboursOp<CPUDevice, float>);
REGISTER_CPU()

#ifdef CUDA
#define EIGEN_USE_GPU
#define REGISTER_GPU()                                                         \
  extern template struct NearestNeighboursFunctor<GPUDevice, float>;           \
  REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device("GPU"),             \
                          NearestNeighboursOp<GPUDevice, float>);
REGISTER_GPU()
#endif

} // namespace functor
} // namespace tensorflow