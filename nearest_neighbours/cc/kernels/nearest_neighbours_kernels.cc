#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "functor.h"

#define EIGEN_USE_THREADS

using namespace Eigen;

class NearestNeighboursOp : public tensorflow::OpKernel {
public:
    explicit NearestNeighboursOp(tensorflow::OpKernelConstruction *context)
            : tensorflow::OpKernel(context) {}

    void Compute(tensorflow::OpKernelContext *context) override {
        // Create inputs
        const tensorflow::Tensor *token_embeddings = nullptr;
        const tensorflow::Tensor *embedding_matrix = nullptr;
        // Check inputs were passed
        OP_REQUIRES_OK(context, context->input("token_embeddings", &token_embeddings));
        OP_REQUIRES_OK(context, context->input("embedding_matrix", &embedding_matrix));

        // Get input dims
        const auto batch_size = static_cast<int32_t>(token_embeddings->dim_size(0));
        const auto vocab_size = static_cast<int32_t>(embedding_matrix->dim_size(0));
        const auto sequence_length = static_cast<int32_t>(token_embeddings->dim_size(1));
        const auto embedding_dim = static_cast<int32_t>(token_embeddings->dim_size(2));
        // Shape Input
        auto embedding_matrix_shaped = embedding_matrix->shaped<float, 2>({vocab_size, embedding_dim});

        // Create output
        tensorflow::Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, token_embeddings->shape(), &output_tensor));
        // Reshape for indexing
        const auto output_shaped = output_tensor->shaped<float, 3>(
                {batch_size, sequence_length, embedding_dim}
        );

        // Convert to Eigen::Matrix for computation
        const auto eigen_embedding_matrix = Map<const Matrix<float, Dynamic, Dynamic, RowMajor>>(
                embedding_matrix->flat<float>().data(), vocab_size, embedding_dim
        );

        // Create thread pool for sharded execution
        auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;

        auto compute_shard = [&sequence_length, &token_embeddings, &vocab_size, &output_shaped, &embedding_dim,
                &eigen_embedding_matrix, &embedding_matrix_shaped](int32_t start, int32_t stop) {

            for (auto batch_index = start; batch_index != stop; batch_index++) {
                const auto sequence = Map<const Matrix<float, Dynamic, Dynamic, RowMajor>>(
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
        };

        // Run sharded kernel
        thread_pool->ParallelFor(batch_size, sequence_length, std::move(compute_shard));
    }
};

REGISTER_KERNEL_BUILDER(Name("NearestNeighbours").Device(tensorflow::DEVICE_CPU), NearestNeighboursOp)