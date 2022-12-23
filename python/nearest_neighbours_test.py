from __future__ import annotations


import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops

nearest_neighbours = load_library.load_op_library(
    resource_loader.get_path_to_datafile("../build/_nearest_neighbours_ops.so")
).nearest_neighbours


@tf.function
def py_nearest_neighbour_single_point(
    token_embedding: tf.Tensor, embedding_matrix: tf.Tensor
) -> tf.Tensor:
    dist = tf.linalg.norm(embedding_matrix - token_embedding, axis=-1)
    index = tf.argmin(dist)
    return tf.gather(embedding_matrix, index, axis=0)


def py_nearest_neighbours(
    token_embeddings: tf.Tensor, embedding_matrix: tf.Tensor
) -> tf.Tensor:
    return tf.stack(
        [
            py_nearest_neighbour_single_point(i, embedding_matrix)
            for i in token_embeddings
        ]
    )


def py_nearest_neighbours_batch(
    token_embeddings_batch: tf.Tensor, embedding_matrix: tf.Tensor
) -> tf.Tensor:
    return tf.stack(
        [py_nearest_neighbours(i, embedding_matrix) for i in token_embeddings_batch]
    )


class NearestNeighboursTest(test.TestCase):
    def testNoNoiseAdded(self):
        with self.test_session():
            em = tf.random.uniform(shape=[50, 32])
            x = tf.convert_to_tensor([[em[0], em[0], em[0]], [em[0], em[0], em[0]]])
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


class TestOnGPU(test.TestCase):

    @test_util.run_gpu_only
    def test_on_gpu(self):
        with self.test_session():
            with ops.device("/gpu:0"):
                x = np.asarray([0.81442218, 0.38077085, 0.92314322, 0.90332325, 0.53478226,
                                0.71365777, 0.37635094, 0.95588844, 0.0587169, 0.44392108,
                                0.25226747, 0.61617229, 0.91802603, 0.37726091, 0.31898511,
                                0.33004847, 0.06517786, 0.0923595, 0.83800092, 0.83094365,
                                0.28403633, 0.91779003, 0.76322004, 0.5993864, 0.45915177,
                                0.00732457, 0.89205795, 0.84037815, 0.94651403, 0.70513769,
                                0.15958934, 0.19663412, 0.56145797, 0.72866532, 0.36341343,
                                0.13236558, 0.8048376, 0.29385706, 0.86819838, 0.84251614,
                                0.33848841, 0.34498596, 0.54794891, 0.83885307, 0.00807368,
                                0.10109079, 0.67029496, 0.57038502, 0.7003306, 0.78164123,
                                0.21382843, 0.44206759, 0.93387193, 0.153378, 0.12464764,
                                0.91483066, 0.23127525, 0.41349707, 0.84495395, 0.36650783,
                                0.76283797, 0.22223185, 0.460718, 0.19223225, 0.4590795,
                                0.27765745, 0.82406614, 0.84014035, 0.29597191, 0.64388382,
                                0.19681925, 0.18043266, 0.94204042, 0.97049734, 0.68002092,
                                0.64209777, 0.83487928, 0.77154018, 0.57515274, 0.09362793,
                                0.38292646, 0.01069972, 0.14793914, 0.07181262, 0.45498261,
                                0.49666209, 0.66090919, 0.84501933, 0.39370952, 0.98664294,
                                0.8860061, 0.08317222, 0.01244135, 0.28049462, 0.92975897,
                                0.37037073, 0.58138978, 0.54119213, 0.76938898, 0.5429423,
                                0.05265366, 0.46610864, 0.2276543, 0.41364711, 0.48217243,
                                0.56950146, 0.83683123, 0.61941795, 0.39421649, 0.31633764,
                                0.84028078, 0.33201169, 0.41313344, 0.44770545, 0.81272186,
                                0.78711288, 0.97535916, 0.98421145, 0.46703603, 0.40114733,
                                0.09849355, 0.73268519, 0.95759776, 0.93780903, 0.7797012,
                                0.51611121, 0.67248324, 0.46384975, 0.25688676, 0.19461875,
                                0.93188473, 0.44659343, 0.69615756, 0.0703793, 0.85710622,
                                0.04640938, 0.58984101, 0.91460141, 0.94151931, 0.71404688,
                                0.72735711, 0.00554461, 0.6317546, 0.09516082, 0.14556217,
                                0.65089767, 0.00558407, 0.34054125, 0.1971325, 0.43612745,
                                0.16133958, 0.85435379, 0.51011338, 0.63289784, 0.29499413,
                                0.24585729, 0.49350811, 0.362212, 0.60338325, 0.56983561]).reshape((8, 4, 5))

                em = np.asarray([0.46822271, 0.25292261, 0.76209782, 0.12390166, 0.1555815,
                                 0.85672186, 0.49593901, 0.08263509, 0.87932451, 0.78133885,
                                 0.2009484, 0.87371986, 0.68930293, 0.7165119, 0.4115712,
                                 0.75252521, 0.57855614, 0.71565441, 0.38461466, 0.57931849,
                                 0.16721307, 0.96660351, 0.64306981, 0.60777359, 0.37837207,
                                 0.16779124, 0.03552278, 0.14275843, 0.82320897, 0.96826605,
                                 0.81388207, 0.74294969, 0.37775112, 0.33168175, 0.49041056,
                                 0.21357541, 0.85828701, 0.42154478, 0.07341114, 0.89896424,
                                 0.33678178, 0.06166559, 0.17811347, 0.27607946, 0.7560883,
                                 0.33443041, 0.8376326, 0.03631953, 0.27371591, 0.32908455,
                                 0.04561242, 0.20201776, 0.85313286, 0.16644364, 0.62910459,
                                 0.65203964, 0.77952139, 0.10028722, 0.53313812, 0.31884443,
                                 0.70770597, 0.62592423, 0.12498556, 0.73189893, 0.93336028,
                                 0.09686865, 0.35148904, 0.95100866, 0.00839949, 0.09237724,
                                 0.40939057, 0.12903606, 0.63592021, 0.8944372, 0.58582662]).reshape((15, 5))
                result = nearest_neighbours(x, em)
                expected = py_nearest_neighbours_batch(x, em)

        self.assertAllClose(result, expected)


if __name__ == "__main__":
    test.main()
