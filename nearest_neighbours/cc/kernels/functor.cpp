//
// Created by Artem Sereda on 15.12.22.
//

#include "functor.h"

using namespace Eigen;


int32_t nearest_neighbour_index(
        const int32_t vocab_size,
        const Vector<float, Dynamic> &embedding,
        const Matrix<float, Dynamic, Dynamic, RowMajor> &embedding_matrix
) {
    auto distances = std::vector<float>(vocab_size);

    const auto embedding_row_major = embedding.transpose();


    for (auto matrix_row_index = 0; matrix_row_index != vocab_size; matrix_row_index++) {
        // Compute distance between current embedding and each matrix row
        const auto row = embedding_matrix.row(matrix_row_index);
        const auto dist = static_cast<float>((row - embedding_row_major).squaredNorm());
        distances[matrix_row_index] = dist;
    }

    // Find index of the smallest distance
    const auto it = std::min_element(std::begin(distances), std::end(distances));
    const auto argmin = static_cast<int32_t>(std::distance(std::begin(distances), it));
    return argmin;
}