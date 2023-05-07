from __future__ import annotations

from typing import Tuple
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.types.core import TensorLike

_backend = load_library.load_op_library(
    resource_loader.get_path_to_datafile("../_nearest_neighbours_op.so")
)

__all__ = ["nearest_neighbours", "nearest_neighbours_indexes"]


def nearest_neighbours(
    token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor
) -> tf.Tensor:
    """
    Take batch of token embeddings, and compute nearest neighbours for each token in Embedding Matrix's space.
    The underlying C++ function expects float32 precision.
    Args:
        token_embeddings:
            A batch of token embeddings with shape [batch_size, None, embedding_dimension].
        embedding_matrix:
            Embedding matrix of Language Model with shape [vocab_size, embedding_dimension].

    Returns:
        token_embeddings, shape = [batch_size, None, embedding_dimension], dtype=tf.float32.
    """
    with tf.name_scope("NearestNeighbours"):
        input_rank, token_embeddings, embedding_matrix = pre_process(token_embeddings, embedding_matrix)
        result = _backend.nearest_neighbours(token_embeddings, embedding_matrix)
        return post_process(input_rank, result)


def nearest_neighbours_indexes(
    token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor
) -> tf.Tensor:
    """
    Take batch of token embeddings, and find index nearest neighbours for each token in Embedding Matrix's space.
    The underlying C++ function expects float32 precision.
    Args:
        token_embeddings:
            A batch of token embeddings with shape [batch_size, None, embedding_dimension].
        embedding_matrix:
            Embedding matrix of Language Model with shape [vocab_size, embedding_dimension].

    Returns:
        indexes, shape = [batch_size, None], dtype=tf.int32.
    """
    with tf.name_scope("NearestNeighboursIndexes"):
        input_rank, token_embeddings, embedding_matrix = pre_process(token_embeddings, embedding_matrix)
        result = _backend.nearest_neighbours_indexes(token_embeddings, embedding_matrix)
        return post_process(input_rank, result)


def pre_process(token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor) -> Tuple[int, tf.Tensor, tf.Tensor]:
    token_embeddings = as_tensor(token_embeddings)
    embedding_matrix = as_tensor(embedding_matrix)
    assert_float32_precision(token_embeddings)
    assert_float32_precision(embedding_matrix)

    em_rank = tf.rank(embedding_matrix)
    if em_rank != 2:
        raise ValueError(f"embedding_matrix must have rank 2, but found {em_rank}")

    input_rank = tf.rank(token_embeddings)
    if input_rank > 3:
        raise ValueError(
            f"token_embeddings can have rank <= 3, but found: {input_rank}"
        )

    return input_rank, token_embeddings, embedding_matrix


def post_process(input_rank: int, result: tf.Tensor) -> tf.Tensor:
    if input_rank == 3:
        return result
    if input_rank == 2:
        return result[0]
    if input_rank == 1:
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
