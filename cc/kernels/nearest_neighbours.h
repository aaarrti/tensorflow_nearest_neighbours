#include "tensorflow/core/platform/types.h"


namespace tensorflow {

  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;
  typedef struct MetalPlugin {} MetalDevice;

  namespace functor {

    template<typename T>
    int32_t nearest_neighbour_index(
        const int32_t vocab_size,
        const Eigen::Vector <T, Eigen::Dynamic> &embedding,
        const Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &embedding_matrix
    );

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

