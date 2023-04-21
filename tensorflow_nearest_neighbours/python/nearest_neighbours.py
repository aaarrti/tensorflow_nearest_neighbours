from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.types.core import TensorLike

_backend = load_library.load_op_library(
    resource_loader.get_path_to_datafile("../_nearest_neighbours_op.so")
)

__all__ = ["nearest_neighbours"]


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
    with tf.name_scope("nearest_neighbours"):
        token_embeddings = as_tensor(token_embeddings)
        embedding_matrix = as_tensor(embedding_matrix)
        assert_float32_precision(token_embeddings)
        assert_float32_precision(embedding_matrix)

        em_rank = tf.rank(embedding_matrix)
        if em_rank != 2:
            raise ValueError(f"embedding_matrix must have rank 2, but found {em_rank}")

        og_rank = tf.rank(token_embeddings)
        if og_rank > 3:
            raise ValueError(
                f"token_embeddings can have rank <= 3, but found: {og_rank}"
            )

        result = _backend.nearest_neighbours(token_embeddings, embedding_matrix)
        if og_rank == 3:
            return result
        if og_rank == 2:
            return result[0]
        if og_rank == 1:
            return result[0, 0]


def as_tensor(x: TensorLike) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return x
    else:
        return tf.constant(x)


def assert_float32_precision(x: tf.Tensor):
    if x.dtype != tf.float32:
        raise ValueError(
            f"underlying C++ implementation expects float32 precision, but found {x.dtype}"
        )
