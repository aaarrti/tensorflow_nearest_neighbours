from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

from tensorflow_nearest_neighbours import nearest_neighbours, nearest_neighbours_indexes


def py_nearest_neighbours(embeddings_batch, embedding_matrix):
    def token_fn(token_embedding):
        dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
        index = tf.argmin(dist)
        return tf.gather(embedding_matrix, index, axis=0)

    def sequence_fn(sentence_embeddings: tf.Tensor):
        return tf.map_fn(token_fn, sentence_embeddings)

    return tf.map_fn(sequence_fn, embeddings_batch)


def py_nearest_neighbours_indexes(embeddings_batch, embedding_matrix):
    def token_fn(token_embedding):
        dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
        index = tf.argmin(dist)
        return tf.cast(index, dtype=tf.float32)

    def sequence_fn(sentence_embeddings: tf.Tensor):
        return tf.map_fn(token_fn, sentence_embeddings)

    return tf.cast(tf.map_fn(sequence_fn, embeddings_batch), dtype=tf.int32)


class TestNearestNeighbours(test.TestCase):
    def testNoNoiseAdded(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.convert_to_tensor([[em[0], em[0], em[0]], [em[0], em[0], em[0]]])
            expected = x
            result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    def testSmallEM(self):
        with self.session():
            with test_util.device(False):
                em = tf.random.uniform(shape=[50, 32])
                x = tf.random.uniform(shape=[8, 10, 32])
                result = nearest_neighbours(x, em)
                expected = py_nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    def testBigEM(self):
        with self.session():
            em = tf.random.uniform(shape=[15000, 512])
            x = tf.random.uniform(shape=[8, 10, 512])
            result = nearest_neighbours(x, em)
            expected = py_nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    def testBigBatch(self):
        with self.session():
            em = tf.random.uniform(shape=[1500, 512])
            x = tf.random.uniform(shape=[32, 65, 512])
            result = nearest_neighbours(x, em)
            expected = py_nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    @test_util.run_gpu_only
    def test_on_gpu(self):
        with self.session():
            with test_util.force_gpu():
                em = tf.random.uniform(shape=[50, 32])
                x = tf.random.uniform(shape=[8, 10, 32])
                result = nearest_neighbours(x, em)
                expected = py_nearest_neighbours(x, em)

        self.assertAllClose(expected, result)


class TestNearestNeighboursIndexes(test.TestCase):
    def testSmallEM(self):
        with self.session():
            with test_util.device(False):
                em = tf.random.uniform(shape=[50, 32])
                x = tf.random.uniform(shape=[8, 10, 32])
                result = nearest_neighbours_indexes(x, em)
                expected = py_nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    def testBigEM(self):
        with self.session():
            em = tf.random.uniform(shape=[15000, 512])
            x = tf.random.uniform(shape=[8, 10, 512])
            result = nearest_neighbours_indexes(x, em)
            expected = py_nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    def testBigBatch(self):
        with self.session():
            em = tf.random.uniform(shape=[1500, 512])
            x = tf.random.uniform(shape=[32, 65, 512])
            result = nearest_neighbours_indexes(x, em)
            expected = py_nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    @test_util.run_gpu_only
    def test_on_gpu(self):
        with self.session():
            with test_util.force_gpu():
                em = tf.random.uniform(shape=[50, 32])
                x = tf.random.uniform(shape=[8, 10, 32])
                result = nearest_neighbours_indexes(x, em)
                expected = py_nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)


if __name__ == "__main__":
    test.main(verbosity=2)
