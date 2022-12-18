#include "external/local_config_tf/include/tensorflow/core/platform/types.h"


namespace tensorflow {
  namespace functor {

    int32_t nearest_neighbour_index(
        const int32_t vocab_size,
        const Eigen::Vector<float, Eigen::Dynamic> &embedding,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &embedding_matrix
    );

    template<typename Device>
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

