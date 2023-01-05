#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace tensorflow::shape_inference;

REGISTER_OP("NearestNeighbours")
        .Input("token_embeddings: float32")
        .Input("embedding_matrix: float32")
        .Output("nearest_neighbours: float32")
        .SetShapeFn([](InferenceContext *c) {
                      ShapeHandle token_embeddings_shape_handle, embedding_matrix_shape_handle;
                      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &token_embeddings_shape_handle));
                      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &embedding_matrix_shape_handle));

                      c->set_output(0, c->MakeShape(
                          {c->Dim(token_embeddings_shape_handle, 0), c->Dim(token_embeddings_shape_handle, 1),
                           c->Dim(token_embeddings_shape_handle, 2)}));
                      return OkStatus();
                    }
        );



