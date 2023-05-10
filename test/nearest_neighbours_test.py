from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from functools import partial
from contextlib import contextmanager

from tensorflow_nearest_neighbours import nearest_neighbours, nearest_neighbours_indexes

tf.config.set_soft_device_placement(True)


@contextmanager
def log_device_placement():
    tf.debugging.set_log_device_placement(True)
    yield
    tf.debugging.set_log_device_placement(False)


def py_nearest_neighbours_1d(token_embedding, embedding_matrix):
    dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
    index = tf.argmin(dist)
    return tf.gather(embedding_matrix, index, axis=0)


def py_nearest_neighbours_2d(sentence_embeddings, embedding_matrix):
    return tf.map_fn(
        partial(py_nearest_neighbours_1d, embedding_matrix=embedding_matrix),
        sentence_embeddings,
    )


def py_nearest_neighbours_3d(embeddings_batch, embedding_matrix):
    return tf.map_fn(
        partial(py_nearest_neighbours_2d, embedding_matrix=embedding_matrix),
        embeddings_batch,
    )


def py_nearest_neighbours_indexes_1d(token_embedding, embedding_matrix, dtype=tf.int32):
    dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
    index = tf.argmin(dist)
    return tf.cast(index, dtype=dtype)


def py_nearest_neighbours_indexes_2d(
    sentence_embeddings, embedding_matrix, dtype=tf.int32
):
    return tf.cast(
        tf.map_fn(
            partial(
                py_nearest_neighbours_indexes_1d,
                embedding_matrix=embedding_matrix,
                dtype=tf.float32,
            ),
            sentence_embeddings,
        ),
        dtype=dtype,
    )


def py_nearest_neighbours_indexes_3d(embeddings_batch, embedding_matrix):
    return tf.cast(
        tf.map_fn(
            partial(
                py_nearest_neighbours_indexes_2d,
                embedding_matrix=embedding_matrix,
                dtype=tf.float32,
            ),
            embeddings_batch,
        ),
        dtype=tf.int32,
    )


class TestNearestNeighboursCPU(test.TestCase):
    def test_1d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[32])
            expected = py_nearest_neighbours_1d(x, em)
            with test_util.device(False):
                with log_device_placement():
                    result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    def test_2d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[10, 32])
            expected = py_nearest_neighbours_2d(x, em)
            with test_util.device(False):
                with log_device_placement():
                    result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    def test_3d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[8, 10, 32])
            expected = py_nearest_neighbours_3d(x, em)
            with test_util.device(False):
                with log_device_placement():
                    result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)


class TestNearestNeighboursGPU(test.TestCase):
    @test_util.run_gpu_only
    def test_1d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[32])
            expected = py_nearest_neighbours_1d(x, em)
            with test_util.device(True):
                with log_device_placement():
                    result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    @test_util.run_gpu_only
    def test_2d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[10, 32])
            expected = py_nearest_neighbours_2d(x, em)
            with test_util.device(True):
                with log_device_placement():
                    result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)

    @test_util.run_gpu_only
    def test_3d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[8, 10, 32])
            expected = py_nearest_neighbours_3d(x, em)
            with test_util.device(True):
                with log_device_placement():
                    result = nearest_neighbours(x, em)

        self.assertAllClose(expected, result)


class TestNearestNeighboursIndexesCPU(test.TestCase):
    def test_1d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[32])
            expected = py_nearest_neighbours_indexes_1d(x, em)
            with test_util.device(False):
                with log_device_placement():
                    result = nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    def test_2d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[10, 32])
            expected = py_nearest_neighbours_indexes_2d(x, em)
            with test_util.device(False):
                with log_device_placement():
                    result = nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    def test_3d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[8, 10, 32])
            expected = py_nearest_neighbours_indexes_3d(x, em)
            with test_util.device(False):
                with log_device_placement():
                    result = nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)


class TestNearestNeighboursIndexesGPU(test.TestCase):
    @test_util.run_gpu_only
    def test_1d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[32])
            expected = py_nearest_neighbours_indexes_1d(x, em)
            with test_util.device(True):
                with log_device_placement():
                    result = nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    @test_util.run_gpu_only
    def test_2d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[10, 32])
            expected = py_nearest_neighbours_indexes_2d(x, em)
            with test_util.device(True):
                with log_device_placement():
                    result = nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)

    @test_util.run_gpu_only
    def test_3d(self):
        with self.session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.random.uniform(shape=[8, 10, 32])
            expected = py_nearest_neighbours_indexes_3d(x, em)
            with test_util.device(True):
                with log_device_placement():
                    result = nearest_neighbours_indexes(x, em)

        self.assertAllEqual(expected, result)


if __name__ == "__main__":
    test.main(verbosity=2)
