#include <iostream>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/types.h"
#include "nearest_neighbours.h"

namespace tensorflow {

  namespace functor {


    class MetalNearestNeighbours {
    private:
      MTL::Device *device;
      MTL::Library *library;
      MTL::Function *function;

      MTL::ComputePipelineState *function_pso;
      MTL::CommandQueue *command_queue;

      MTL::Buffer *m_buffer_token_embeddings;
      MTL::Buffer *m_buffer_embedding_matrix;
      MTL::Buffer *m_buffer_result;

    public:

      void init() {
        if (this->device == nullptr) {
          this.device = MTL::CreateSystemDefaultDevice();
        }

        if (this->library == nullptr) {
          NS::Error *error = nullptr;
          NS::String *path = NS::String::string(
              "/Users/artemsereda/Documents/PycharmProjects/tf_nearest_neighbours/build/_nearest_neighbours_kernel.metallib",
              NS::StringEncoding::ASCIIStringEncoding
          );
          this.library = m_device->newLibrary(path, &error);
          if (!this->library) {
            std::cerr << "Failed to load default library: " << error->localizedDescription()->utf8String() << std::endl;
            abort();
          }

        }

        if (this->function == nullptr) {
          NS::String *function_name = NS::String::string("nearest_neighbours", NS::ASCIIStringEncoding);
          this.function = default_library->newFunction(function_name);
          if (!this.function) {
            std::cerr << "Failed to find the adder function." << std::endl;
            abort();
          }
        }


        auto function_name = NS::String::string("nearest_neighbours", NS::ASCIIStringEncoding);
        auto add_function = default_library->newFunction(function_name);

        if (!this->function) {
          std::cerr << "Failed to find the adder function." << error->localizedDescription()->utf8String() << std::endl;
        }
        this->function_pso = m_device->newComputePipelineState(this.function, &error);
        this->command_queue = m_device->newCommandQueue();
      }

      void load_data_to_buffers(
          //floa* token_embeddings,
          //T* embedding_matrix,
          const int32_t &sequence_length,
          const int32_t &vocab_size,
          const int32_t &embed_dim,
      ) {
        this-> = m_device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
        m_buffer_B = m_device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
        m_buffer_result = m_device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);

        generate_random_float_data(m_buffer_A);
        generate_random_float_data(m_buffer_B);

        float *data_ptr = (float *) buffer->contents();
        for (unsigned long index = 0; index < array_length; index++) {
          data_ptr[index] = (float) rand() / (float) (RAND_MAX);
        }
      }

      void compute() {
        MTL::CommandBuffer *command_buffer = this->command_queue->commandBuffer();
        MTL::ComputeCommandEncoder *compute_encoder = this->command_buffer->computeCommandEncoder();
        compute_encoder->setComputePipelineState(m_add_function_pso);
        compute_encoder->setBuffer(m_buffer_A, 0, 0);
        compute_encoder->setBuffer(m_buffer_B, 0, 1);
        compute_encoder->setBuffer(m_buffer_result, 0, 2);

        MTL::Size grid_size = MTL::Size(array_length, 1, 1);

        NS::UInteger thread_group_size_ = m_add_function_pso->maxTotalThreadsPerThreadgroup();
        if (thread_group_size_ > array_length) {
          thread_group_size_ = array_length;
        }

        MTL::Size thread_group_size = MTL::Size(thread_group_size_, 1, 1);

        compute_encoder->dispatchThreads(grid_size, thread_group_size);
        compute_encoder->endEncoding();
        command_buffer->commit();
        command_buffer->waitUntilCompleted();

      };


    }

    template typename <T>
    struct NearestNeighboursFunctor<MetalDevice, T> {

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

        const T *token_embeddings_flat = token_embeddings->flat < T >.data();
        const T *embedding_matrix_flat = embedding_matrix->flat<T>().data();
        T *output_flat = output_tensor->flat<T>().data();

        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

        const m_nn = std::make_unique<MetalNearestNeighbours>();
        m_nn->init();
        m_nn->load_data_to_buffers<T>(
            sequence_length, vocab_size, embedding_dim,
            token_embeddings_flat, embedding_matrix_flat, output_flat
        );
        m_nn->compute();
        p_pool->release();
      }

    };
  }
}

