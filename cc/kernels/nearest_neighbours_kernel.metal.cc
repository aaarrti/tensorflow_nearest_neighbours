#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <iostream>
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "nearest_neighbours.h"


namespace tensorflow {

  template<>
  const Eigen::MetalDevice &OpKernelContext::eigen_device() const {
    static const Eigen::MetalDevice device{};
    return device;
  }

  namespace functor {

    namespace metal {

        template<typename T>
      MTL::Buffer *create_buffer(MTL::Device* m_device, const T *arr, int size) {
        const auto buffer = m_device->newBuffer(size, MTL::ResourceStorageModeShared);
        auto *data_ptr = (__fp16 *) buffer->contents();
        for (auto index = 0; index < size; index++) {
          std::cout << index << std::endl;
          data_ptr[index] = (__fp16) arr[index];
        }
        return buffer;
      }
    }

    template<typename T>
    struct NearestNeighboursFunctor<MetalDevice, T> {

      void operator()(
          const MetalDevice &device,
          const Tensor *token_embeddings,
          const Tensor *embedding_matrix,
          Tensor *output_tensor
      ) {

        const auto batch_size = static_cast<int32_t>(token_embeddings->dim_size(0));
        const auto vocab_size = static_cast<int32_t>(embedding_matrix->dim_size(0));
        const auto num_tokens = static_cast<int32_t>(token_embeddings->dim_size(1));
        const auto embedding_dim = static_cast<int32_t>(token_embeddings->dim_size(2));

        const T* token_embeddings_flat = (T*) token_embeddings->data();
        const T* embedding_matrix_flat = (T*) embedding_matrix->data();
        T* output_flat = (T*) output_tensor->data();


        NS::AutoreleasePool *p_pool = NS::AutoreleasePool::alloc()->init();
        MTL::Device *m_device = MTL::CreateSystemDefaultDevice();
        NS::Error *error = nullptr;
        NS::String *path = NS::String::string("_nearest_neighbours_kernel.metallib", NS::StringEncoding::ASCIIStringEncoding);
        auto library = m_device->newLibrary(path, &error);
        if (!library) {
          std::cerr << "Failed to load default library: " << error->localizedDescription()->utf8String() << std::endl;
          abort();
        }

        const auto function_name = NS::String::string("nearest_neighbours", NS::ASCIIStringEncoding);
        const auto function = library->newFunction(function_name);

        if (!function) {
          std::cerr << "Failed to find the adder function." << std::endl;
        }

        const auto function_pso = m_device->newComputePipelineState(function, &error);
        if (error) {
          std::cerr << error->localizedDescription()->utf8String() << std::endl;
          abort();
        }

        const auto m_command_queue = m_device->newCommandQueue();
        const auto m_buffer_X = metal::create_buffer<T>(m_device, token_embeddings_flat, batch_size * num_tokens * embedding_dim);
        std::cout << "-----Metal Kernel ----" << std::endl;
        const auto m_buffer_EM = metal::create_buffer<T>(m_device, embedding_matrix_flat, vocab_size * embedding_dim);
        const auto m_buffer_result = m_device->newBuffer(batch_size * num_tokens * embedding_dim, MTL::ResourceStorageModeShared);



        MTL::CommandBuffer *command_buffer = m_command_queue->commandBuffer();
        MTL::ComputeCommandEncoder *compute_encoder = command_buffer->computeCommandEncoder();

        compute_encoder->setComputePipelineState(function_pso);
        compute_encoder->setBuffer(m_buffer_X, 0, 0);
        compute_encoder->setBuffer(m_buffer_EM, 0, 1);
        compute_encoder->setBuffer(m_buffer_result, 0, 2);
        compute_encoder->setBytes(&num_tokens, sizeof(int16_t), 3);
        compute_encoder->setBytes(&vocab_size, sizeof(int16_t), 4);
        compute_encoder->setBytes(&embedding_dim, sizeof(int16_t), 5);

        const auto grid_size = MTL::Size(batch_size, num_tokens, 1);
        const auto thread_group_size = MTL::Size(batch_size, num_tokens, 1);
        compute_encoder->dispatchThreads(grid_size, thread_group_size);
        compute_encoder->endEncoding();
        command_buffer->commit();
        command_buffer->waitUntilCompleted();

        auto *data_ptr = (__fp16 *) m_buffer_result->contents();
        for (int i = 0; i != batch_size * num_tokens * embedding_dim; i++) {
          output_flat[i] = (T) data_ptr[i];
        }
      }
    };

    template
    struct NearestNeighboursFunctor<MetalDevice, float>;
  }
}


