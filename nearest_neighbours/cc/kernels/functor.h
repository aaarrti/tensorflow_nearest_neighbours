#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;

int32_t nearest_neighbour_index(
        const int32_t vocab_size,
        const Vector<float, Dynamic> &embedding,
        const Matrix<float, Dynamic, Dynamic, RowMajor> &embedding_matrix
);