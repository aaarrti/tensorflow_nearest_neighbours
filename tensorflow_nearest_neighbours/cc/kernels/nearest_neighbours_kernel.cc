#define EIGEN_USE_THREADS

#include "nearest_neighbours.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/tsl/platform/logging.h"

namespace tensorflow {

  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;

  namespace functor {

    namespace {

      inline __attribute__((always_inline)) int index_2d_flat(const int index_0, const int index_1, const int shape_1) {
        return index_1 + index_0 * shape_1;
      }

      inline __attribute__((always_inline)) int index_3d_flat(const int index_0, const int index_1, const int index_2,
                                                              const int shape_1, const int shape_2) {
        return index_2 + index_1 * shape_2 + index_0 * shape_2 * shape_1;
      }


      template<typename T>
      requires std::floating_point<T>
      int nearest_neighbour_index(
        int vocab_size,
        const Eigen::Vector<T, Eigen::Dynamic> embedding,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> embedding_matrix
      ) {
        auto distances = std::vector<T>(vocab_size);
        const auto embedding_row_major = embedding.transpose();

        for (auto matrix_row_index = 0; matrix_row_index != vocab_size; matrix_row_index++) {
          // Compute distance between current embedding and each matrix row
          const auto row = embedding_matrix.row(matrix_row_index);
          const auto dist = static_cast<T>((row - embedding_row_major).squaredNorm());
          distances[matrix_row_index] = dist;
        }

        // Find index of the smallest distance
        const auto it = std::min_element(std::begin(distances), std::end(distances));
        const auto argmin = static_cast<int>(std::distance(std::begin(distances), it));
        return argmin;
      }
    }

    // --------------------------------------------------------------------------------------------------

    template<typename T> requires std::floating_point<T>
    struct NearestNeighboursIndexesFunctor<1, CPUDevice, T> {
      void operator()(
        const CPUDevice &d,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        int *output
      ) {

        const auto eigen_embedding_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          embedding_matrix, vocab_size, embedding_dim
        );
        // input is 1D.
        const auto eigen_token_embeddings = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(
          token_embeddings, embedding_dim
        );

        auto argmin = nearest_neighbour_index<T>(vocab_size, eigen_token_embeddings, eigen_embedding_matrix);
        // output is scalar.
        output[0] = argmin;
      }
    };

    template<typename T> requires std::floating_point<T>
    struct NearestNeighboursIndexesFunctor<2, CPUDevice, T> {
      void operator()(
        const CPUDevice &device,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        int *output
      ) {

        const auto eigen_embedding_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          embedding_matrix, vocab_size, embedding_dim
        );
        // input is 2D.
        const auto eigen_token_embeddings = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          token_embeddings, num_tokens, embedding_dim
        );

        device.parallelFor(
          num_tokens,
          Eigen::TensorOpCost{
            static_cast<double>(vocab_size * sizeof(float)),
            static_cast<double>(vocab_size * sizeof(float)),
            static_cast<double>(vocab_size)
          },
          [
            vocab_size,
            output,
            eigen_token_embeddings,
            eigen_embedding_matrix
          ](int start, int stop) {
            for (auto token_index = start; token_index != stop; token_index++) {

              const auto eigen_embedding = eigen_token_embeddings.row(token_index);

              // Find index of the smallest distance
              auto argmin = nearest_neighbour_index<T>(vocab_size, eigen_embedding, eigen_embedding_matrix);
              // Output is 1D.
              output[token_index] = argmin;
            }
          }
        );
      }
    };

    template<typename T> requires std::floating_point<T>
    struct NearestNeighboursIndexesFunctor<3, CPUDevice, T> {
      void operator()(
        const CPUDevice &device,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        int *output
      ) {

        // Convert to Eigen::Matrix for computation
        const auto eigen_embedding_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          embedding_matrix, vocab_size, embedding_dim
        );

        device.parallelFor(
          batch_size,
          Eigen::TensorOpCost{
            static_cast<double>(batch_size * vocab_size * sizeof(float)),
            static_cast<double>(batch_size * vocab_size * sizeof(float)),
            static_cast<double>(batch_size * vocab_size)
          },
          [
            eigen_embedding_matrix,
            token_embeddings,
            num_tokens,
            embedding_dim,
            vocab_size,
            output
          ](int start, int stop) {
            for (auto batch_index = start; batch_index != stop; batch_index++) {
              const auto sequence = Eigen::Map<
                const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
              >(token_embeddings + index_3d_flat(batch_index, 0, 0, num_tokens, embedding_dim),
                vocab_size, embedding_dim
              );

              for (auto token_index = 0; token_index != num_tokens; token_index++) {

                const auto embedding = sequence.row(token_index);

                // Find index of the smallest distance
                auto argmin = nearest_neighbour_index<T>(vocab_size, embedding, eigen_embedding_matrix);
                // Output is 2D.
                output[index_2d_flat(batch_index, token_index, num_tokens)] = argmin;

              }
            }
          }
        );
      }
    };


    // --------------------------------------------------------------------------------------------------


    template<typename T> requires std::floating_point<T>
    struct NearestNeighboursFunctor<1, CPUDevice, T> {
      void operator()(
        const CPUDevice &device,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        T *output
      ) {
        const auto eigen_embedding_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          embedding_matrix, vocab_size, embedding_dim
        );
        const auto eigen_token_embeddings = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(
          token_embeddings, embedding_dim
        );

        auto argmin = nearest_neighbour_index<T>(vocab_size, eigen_token_embeddings, eigen_embedding_matrix);
        for (auto i = 0; i != embedding_dim; i++) {
          // output is 1D.
          output[i] = embedding_matrix[index_2d_flat(argmin, i, embedding_dim)];
        }
      }
    };


    template<typename T> requires std::floating_point<T>
    struct NearestNeighboursFunctor<2, CPUDevice, T> {
      void operator()(
        const CPUDevice &device,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        T *output
      ) {

        const auto eigen_embedding_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          embedding_matrix, vocab_size, embedding_dim
        );

        const auto eigen_token_embeddings = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          token_embeddings, num_tokens, embedding_dim
        );


        device.parallelFor(
          num_tokens,
          Eigen::TensorOpCost{
            static_cast<double>(vocab_size * sizeof(float)),
            static_cast<double>(vocab_size * sizeof(float)),
            static_cast<double>(vocab_size)
          },
          [
            eigen_token_embeddings,
            eigen_embedding_matrix,
            embedding_dim,
            output,
            vocab_size,
            embedding_matrix
          ](int start, int stop) {

            for (auto token_index = start; token_index != stop; token_index++) {
              const auto eigen_embedding = eigen_token_embeddings.row(token_index);

              // Find index of the smallest distance
              auto argmin = nearest_neighbour_index<T>(vocab_size, eigen_embedding, eigen_embedding_matrix);

              // Fill the output
              for (auto i = 0; i != embedding_dim; i++) {
                output[index_2d_flat(token_index, i, embedding_dim)] =
                  embedding_matrix[index_2d_flat(argmin, i, embedding_dim)];
              }
            }
          }
        );

      }

    };


    template<typename T> requires std::floating_point<T>
    struct NearestNeighboursFunctor<3, CPUDevice, T> {
      void operator()(
        const CPUDevice &device,
        const int batch_size,
        const int num_tokens,
        const int vocab_size,
        const int embedding_dim,
        const T *token_embeddings,
        const T *embedding_matrix,
        T *output
      ) {

        // Convert to Eigen::Matrix for computation
        const auto eigen_embedding_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          embedding_matrix, vocab_size, embedding_dim
        );


        device.parallelFor(
          batch_size,
          Eigen::TensorOpCost{
            static_cast<double>(batch_size * vocab_size * sizeof(float)),
            static_cast<double>(batch_size * vocab_size * sizeof(float)),
            static_cast<double>(batch_size * vocab_size)
          },
          [
            token_embeddings,
            embedding_matrix,
            eigen_embedding_matrix,
            vocab_size,
            embedding_dim,
            num_tokens,
            output
          ](int start, int stop) {
            for (auto batch_index = start; batch_index != stop; batch_index++) {
              const auto sequence = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                token_embeddings + index_3d_flat(batch_index, 0, 0, num_tokens, embedding_dim),
                vocab_size, embedding_dim
              );

              for (auto token_index = 0; token_index != num_tokens; token_index++) {
                auto distances = std::vector<T>(vocab_size);

                const auto embedding = sequence.row(token_index);

                // Find index of the smallest distance
                auto argmin = nearest_neighbour_index<T>(vocab_size, embedding, eigen_embedding_matrix);

                // Fill the output
                for (auto i = 0; i != embedding_dim; i++) {
                  output[index_3d_flat(batch_index, token_index, i, num_tokens, embedding_dim)] =
                    embedding_matrix[index_2d_flat(argmin, i, embedding_dim)];
                }
              }
            }
          }
        );

      }
    };

    template<typename Device, typename T> requires std::floating_point<T>
    class NearestNeighboursIndexesOp : public tensorflow::OpKernel {
    public:
      explicit NearestNeighboursIndexesOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}

      void Compute(tensorflow::OpKernelContext *context) override {
        // Create inputs
        const tensorflow::Tensor *token_embeddings = nullptr;
        const tensorflow::Tensor *embedding_matrix = nullptr;
        // Check inputs were passed
        OP_REQUIRES_OK(context, context->input("token_embeddings", &token_embeddings));
        OP_REQUIRES_OK(context, context->input("embedding_matrix", &embedding_matrix));
        // Create output
        tensorflow::Tensor *output_tensor = nullptr;
        const auto ndim = embedding_matrix->shape().num_elements();
        const auto vocab_size = static_cast<int>(embedding_matrix->dim_size(0));
        const auto embedding_dim = static_cast<int>(token_embeddings->dim_size(2));

        switch (ndim) {
          case 1: {
            OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output_tensor));
            NearestNeighboursIndexesFunctor<1, Device, T>()(
              context->eigen_device<Device>(),
              0,
              0,
              vocab_size,
              embedding_dim,
              token_embeddings->flat<T>().data(),
              embedding_matrix->flat<T>().data(),
              output_tensor->flat<int>().data()
            );
            break;
          }
          case 2: {
            const auto num_tokens = static_cast<int>(token_embeddings->dim_size(0));
            OP_REQUIRES_OK(context, context->allocate_output(0, {num_tokens}, &output_tensor));
            NearestNeighboursIndexesFunctor<2, Device, T>()(
              context->eigen_device<Device>(),
              0,
              num_tokens,
              vocab_size,
              embedding_dim,
              token_embeddings->flat<T>().data(),
              embedding_matrix->flat<T>().data(),
              output_tensor->flat<int>().data()
            );
            break;
          }
          case 3: {
            const auto batch_size = static_cast<int>(token_embeddings->dim_size(0));
            const auto num_tokens = static_cast<int>(token_embeddings->dim_size(1));
            OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size, num_tokens}, &output_tensor));
            NearestNeighboursIndexesFunctor<3, Device, T>()(
              context->eigen_device<Device>(),
              batch_size,
              num_tokens,
              vocab_size,
              embedding_dim,
              token_embeddings->flat<T>().data(),
              embedding_matrix->flat<T>().data(),
              output_tensor->flat<int>().data()
            );
          }
          default:
            break;
        }
      }

    };

    template<typename Device, typename T> requires std::floating_point<T>
    class NearestNeighboursOp : public tensorflow::OpKernel {
    public:
      explicit NearestNeighboursOp(tensorflow::OpKernelConstruction *context) : OpKernel(context) {}

      void Compute(tensorflow::OpKernelContext *context) override {
        // Create inputs
        const tensorflow::Tensor *token_embeddings = nullptr;
        const tensorflow::Tensor *embedding_matrix = nullptr;
        // Check inputs were passed
        OP_REQUIRES_OK(context, context->input("token_embeddings", &token_embeddings));
        OP_REQUIRES_OK(context, context->input("embedding_matrix", &embedding_matrix));
        // Create output
        tensorflow::Tensor *output_tensor = nullptr;
        const auto ndim = embedding_matrix->shape().num_elements();
        const auto vocab_size = static_cast<int>(embedding_matrix->dim_size(0));
        const auto embedding_dim = static_cast<int>(token_embeddings->dim_size(2));

        switch (ndim) {
          case 1: {
            OP_REQUIRES_OK(context, context->allocate_output(0, {embedding_dim}, &output_tensor));
            NearestNeighboursFunctor<1, Device, T>()(
              context->eigen_device<Device>(),
              0,
              0,
              vocab_size,
              embedding_dim,
              token_embeddings->flat<T>().data(),
              embedding_matrix->flat<T>().data(),
              output_tensor->flat<T>().data()
            );
            break;
          }
          case 2: {
            const auto num_tokens = static_cast<int>(token_embeddings->dim_size(0));
            OP_REQUIRES_OK(context, context->allocate_output(0, {num_tokens, embedding_dim}, &output_tensor));
            NearestNeighboursFunctor<2, Device, T>()(
              context->eigen_device<Device>(),
              0,
              num_tokens,
              vocab_size,
              embedding_dim,
              token_embeddings->flat<T>().data(),
              embedding_matrix->flat<T>().data(),
              output_tensor->flat<T>().data()
            );
            break;
          }
          case 3: {
            const auto batch_size = static_cast<int>(token_embeddings->dim_size(0));
            const auto num_tokens = static_cast<int>(token_embeddings->dim_size(1));
            OP_REQUIRES_OK(
              context,
              context->allocate_output(0, {batch_size, num_tokens, embedding_dim}, &output_tensor)
            );
            NearestNeighboursFunctor<3, Device, T>()(
              context->eigen_device<Device>(),
              batch_size,
              num_tokens,
              vocab_size,
              embedding_dim,
              token_embeddings->flat<T>().data(),
              embedding_matrix->flat<T>().data(),
              output_tensor->flat<T>().data()
            );
            break;
          }
          default:
            break;
        }
      }
    };


    REGISTER_KERNEL_BUILDER(Name("NearestNeighboursIndexes").Device("CPU"),
                            NearestNeighboursIndexesOp<CPUDevice, float>);
    REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device("CPU"), NearestNeighboursOp<CPUDevice, float>);


#ifdef CUDA
#define REGISTER_GPU()                                                         \
  extern template struct NearestNeighboursFunctor<GPUDevice, float>;           \
  REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device("GPU"),             \
                          NearestNeighboursOp<GPUDevice, float>);
    REGISTER_GPU()
#endif

  } // namespace functor
} // namespace tensorflow