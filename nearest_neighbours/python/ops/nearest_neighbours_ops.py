from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader




_backend = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_nearest_neighbours_ops.so")
)


def nearest_neighbours(
    token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor
) -> tf.Tensor:
    """
    Take batch of token embeddings, and compute nearest neighbours for each token in Embedding Matrix's space.
    The underlying C++ function expects float32 precision.

    :param token_embeddings: A batch of token embeddings with shape [batch_size, None, embedding_dimension].
    :param embedding_matrix: Embedding matrix of Language Model with shape [vocab_size, embedding_dimension].
    :return: token_embeddings, shape = [batch_size, None, embedding_dimension], dtype=tf.float32.
    """
    return _backend.nearest_neighbours(token_embeddings, embedding_matrix)
