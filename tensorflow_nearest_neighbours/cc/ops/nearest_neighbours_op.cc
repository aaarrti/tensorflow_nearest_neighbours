#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("NearestNeighbours")
    .Input("token_embeddings: float32")
    .Input("embedding_matrix: float32")
    .Output("nearest_neighbours: float32")
    .SetShapeFn(
      [](InferenceContext *c) {
        ShapeHandle token_embeddings_shape_handle, embedding_matrix_shape_handle;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &embedding_matrix_shape_handle));
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 3, &token_embeddings_shape_handle));
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &token_embeddings_shape_handle));


        auto rank = InferenceContext::Rank(c->input(0));
        std::vector<DimensionHandle> dims;
        dims.push_back(c->Dim(c->input(0), 0));
        if (rank > 1) {
          dims.push_back(c->Dim(c->input(0), 1));
        }
        if (rank > 2) {
          dims.push_back(c->Dim(c->input(0), 2));
        }
        c->set_output(0, c->MakeShape(dims));
        return OkStatus();
      }
    );


REGISTER_OP("NearestNeighboursIndexes")
    .Input("token_embeddings: float32")
    .Input("embedding_matrix: float32")
    .Output("nearest_neighbours_indexes: int32")
    .SetShapeFn(
      [](InferenceContext *c) {
        ShapeHandle token_embeddings_shape_handle, embedding_matrix_shape_handle;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &embedding_matrix_shape_handle));
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 3, &token_embeddings_shape_handle));
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &token_embeddings_shape_handle));

        auto rank = InferenceContext::Rank(c->input(0));
        if (rank == 1) {
          c->set_output(0, c->Scalar());
          return OkStatus();
        }
        std::vector<DimensionHandle> dims;
        dims.push_back(c->Dim(c->input(0), 0));
        if (rank > 2) {
          dims.push_back(c->Dim(c->input(0), 1));
        }
        c->set_output(0, c->MakeShape(dims));
        return OkStatus();
      }
    );