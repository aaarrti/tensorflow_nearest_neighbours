# TensorFlow Nearest Neighbours Op

## Example usage:

```python
import tensorflow as tf
from nearest_neighbours import nearest_neighbours

tf.debugging.set_log_device_placement(True)

word_embeddings_batch = tf.random.uniform(shape=[8, 10, 32])
embedding_matrix = tf.random.uniform(shape=[500, 32])
nearest_neighbours(word_embeddings_batch, embedding_matrix)
```

### Tools and version used to build binaries:

| Tool       | Ubuntu    | MacOS     |
|------------|-----------|-----------|
| OS         | 20.04.5   | 12.6.1    |
| Clang      | 10.0.0    | 14.0.0    |
| Tensorflow | 2.11.0    | 2.11.0    | 
| Python     | 3.9       | 3.8       |
| cuda       | 11.2      | -         | 
| nvcc       | V11.2.152 | -         | 
| metal      | -         | 31001.667 | 
| metallib   | -         | 31001.667 |      
 | bazel      | 6.0.0     | 6.0.0     |

### Building from source:
```bash
./configure
bazelisk run build_pip_pkg
```
### Running tests:
```bash
bazelisk run //nearest_neighbours:nearest_neighbours_py_test 
```

