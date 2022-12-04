from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

nearest_neighbours_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_nearest_neighbours_ops.so')
)

nearest_neighbours_func = nearest_neighbours_ops.nearest_neighbours
nearest_neighbours_indexes_func = nearest_neighbours_ops.nearest_neighbours_indexes


__all__ = [
    "nearest_neighbours",
    "nearest_neighbours_indexes",
]


def nearest_neighbours(token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    """
    Take batch of token embeddings, and compute nearest neighbours for each token in Embedding Matrix's space.
    The underlying C++ function expects float32 precision.

    :param token_embeddings: A batch of token embeddings with shape [batch_size, None, embedding_dimension].
    :param embedding_matrix: Embedding matrix of Language Model with shape [vocab_size, embedding_dimension].
    :return: token_embeddings, shape = [batch_size, None, embedding_dimension], dtype=tf.float32.
    """
    return nearest_neighbours_func(token_embeddings, embedding_matrix)


def nearest_neighbours_indexes(token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    """
    Take batch of token embeddings, and compute indexes (words) of nearest neighbours for each token in Embedding
    Matrix's space. The underlying C++ function expects float32 precision.

    :param token_embeddings: A batch of token embeddings with shape [batch_size, None, embedding_dimension].
    :param embedding_matrix: Embedding matrix of Language Model with shape [vocab_size, embedding_dimension].
    :return: indexes, shape = [batch_size, None], dtype=tf.int32
    """
    return nearest_neighbours_indexes_func(token_embeddings, embedding_matrix)