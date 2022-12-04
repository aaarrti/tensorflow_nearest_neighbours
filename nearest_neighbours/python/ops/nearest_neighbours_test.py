from __future__ import annotations

import tensorflow as tf
from tensorflow.python.platform import test

try:
    from nearest_neighbours.python.ops.nearest_neighbours_ops import nearest_neighbours, \
        nearest_neighbours_indexes
except ImportError:
    from nearest_neighbours_ops import nearest_neighbours, nearest_neighbours_indexes


@tf.function
def py_nearest_neighbour_single_point(token_embedding: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
    index = tf.argmin(dist)
    return tf.gather(embedding_matrix, index, axis=0)


def py_nearest_neighbours(token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        py_nearest_neighbour_single_point(i, embedding_matrix)
        for i in token_embeddings
    ])


def py_nearest_neighbours_batch(token_embeddings_batch: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        py_nearest_neighbours(i, embedding_matrix)
        for i in token_embeddings_batch
    ])


class NearestNeighboursTest(test.TestCase):

    def testNoNoiseAdded(self):
        with self.test_session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.convert_to_tensor([
                [em[0], em[0], em[0]],
                [em[0], em[0], em[0]]
            ])
            expected = x
            result = nearest_neighbours(x, em)
            self.assertAllClose(result, expected)

    def testSmallEM(self):
        with self.test_session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[8, 10, 32])

            result = nearest_neighbours(x, em)
            expected = py_nearest_neighbours_batch(x, em)

            self.assertAllClose(result, expected)

    def testBigEM(self):
        with self.test_session():
            em = tf.random.uniform(shape=[15000, 512])
            x = tf.random.uniform(shape=[8, 10, 512])

            result = nearest_neighbours(x, em)
            expected = py_nearest_neighbours_batch(x, em)

            self.assertAllClose(result, expected)

    def testBigBatch(self):
        with self.test_session():
            em = tf.random.uniform(shape=[1500, 512])
            x = tf.random.uniform(shape=[32, 65, 512])

            result = nearest_neighbours(x, em)
            expected = py_nearest_neighbours_batch(x, em)

        self.assertAllClose(result, expected)


@tf.function
def py_nearest_neighbour_index(token_embedding: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
    index = tf.argmin(dist)
    return index


def py_nearest_neighbours_indexes(token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        py_nearest_neighbour_index(i, embedding_matrix)
        for i in token_embeddings
    ])


def py_nearest_neighbours_indexes_batch(token_embeddings_batch: tf.Tensor, embedding_matrix: tf.Tensor) -> tf.Tensor:
    return tf.stack([
        py_nearest_neighbours_indexes(i, embedding_matrix)
        for i in token_embeddings_batch
    ])


class NearestNeighboursIndexesTest(test.TestCase):

    def testNoNoiseAdded(self):
        with self.test_session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.convert_to_tensor([
                [em[0], em[0], em[0]],
                [em[0], em[0], em[0]]
            ])
            expected = tf.convert_to_tensor([
                [0, 0, 0],
                [0, 0, 0]
            ])
            result = nearest_neighbours_indexes(x, em)
            self.assertAllClose(result, expected)

    def testSmallEM(self):
        with self.test_session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[8, 10, 32])

            result = nearest_neighbours_indexes(x, em)
            expected = py_nearest_neighbours_indexes_batch(x, em)

            self.assertAllClose(result, expected)

    def testBigEM(self):
        with self.test_session():
            em = tf.random.uniform(shape=[15000, 512])
            x = tf.random.uniform(shape=[8, 10, 512])

            result = nearest_neighbours_indexes(x, em)
            expected = py_nearest_neighbours_indexes_batch(x, em)

            self.assertAllClose(result, expected)

    def testBigBatch(self):
        with self.test_session():
            em = tf.random.uniform(shape=[1500, 512])
            x = tf.random.uniform(shape=[32, 65, 512])

            result = nearest_neighbours_indexes(x, em)
            expected = py_nearest_neighbours_indexes_batch(x, em)

            self.assertAllClose(result, expected)


if __name__ == '__main__':
    test.main()
