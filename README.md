# TensorFlow Nearest Neighbours Op

| Tool       | Version    |
|------------|------------|
| Bazel      | 5.1.1      |
| Clang      | 14.0.0     |
| Tensorflow | 2.10.0-cpu |
| Python     | 3.9        |
| Protobuf   | 3.19.6     |

### Run Tests

```bash
bazel run //nearest_neighbours:nearest_neighbours_ops_py_test       
```

### Build PIP Package

```bash
  ./configure.sh
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
```

### Install and Test PIP Package

Once the pip package has been built, you can install it with

```bash
pip install artifacts/*.whl
```

Then test out the pip package

```python
import tensorflow as tf
from nearest_neighbours import nearest_neighbours

x = tf.random.uniform(shape=[8, 10, 32])
em = tf.random.uniform(shape=[500, 32])
result = nearest_neighbours(x, em)
print(result.shape)
```

## References:

- [Extending TensorFlow with Custom C++ Operations](https://www.gresearch.co.uk/blog/article/extending-tensorflow-with-custom-c-operations/)
- [Create an Op](https://www.tensorflow.org/guide/create_op)
- [TensorFlow Custom Op](https://github.com/tensorflow/custom-op)

## TODO:

- Add GPU (Metal) Implementation
  Reference: [Customizing a TensorFlow operation](https://developer.apple.com/documentation/metal/metal_sample_code_library/customizing_a_tensorflow_operation)
  
