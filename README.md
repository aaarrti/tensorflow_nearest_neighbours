# TensorFlow Nearest Neighbours Op
Given an embedding matrix `EM`, and batch of word embeddings `x` 
find nearest embedding for each token `x_ij` in `EM`.

```python
import tensorflow as tf
from tensorflow_nearest_neighbours import nearest_neighbours
tf.debugging.set_log_device_placement(True)

x = tf.random.uniform(shape=[8, 10, 32])
EM = tf.random.uniform(shape=[500, 32])
result = nearest_neighbours(x, EM)
print(result.shape)
```

#### Instalation
```shell
pip install tensorflow_nearest_neighbours
```
