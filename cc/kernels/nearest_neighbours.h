namespace Eigen {
  typedef struct MetalPlugin {} MetalDevice;
}

namespace tensorflow {

  typedef Eigen::ThreadPoolDevice CPUDevice;
  typedef Eigen::GpuDevice GPUDevice;
  typedef Eigen::MetalDevice MetalDevice;

  namespace functor {

    template<typename Device, typename T>
    struct NearestNeighboursFunctor {
      void operator()(
          const Device &d,
          const Tensor *token_embeddings,
          const Tensor *embedding_matrix,
          Tensor *output_tensor
      );
    };
  }
}

