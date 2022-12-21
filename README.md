# TensorFlow Nearest Neighbours Op


| Tool         | Version |
|--------------|---------|
| Clang        | 14.0.0  |
| Tensorflow   | 2.11.0  |
| Python       | 3.9     |
 | gcc          |         |
| cuda         |         | 
| nvcc         |         | 
| metal        |         | 
| metallib     |         |

### Example usage:

```python
import tensorflow as tf
from nearest_neighbours import nearest_neighbours

x = tf.random.uniform(shape=[8, 10, 32])
em = tf.random.uniform(shape=[500, 32])
result = nearest_neighbours(x, em)
print(result.shape)
```

### Building from source:
- First we need to build the shared library
  - CPU only:
    ```bash
    make cpu_kernel
    ```
  - CUDA 
    ```bash
    make cuda_kernel
    ```
  - Metal (macOS only)
    ```bash
    make metal_kernel
    ```
    
- Then, we can test the OP
```bash
make test
```
- Afterwards, we build a pip package from it:
```bash
make pip_pkg
```
- And finally, we can install it: 
```bash
pip install build/dist/*.whl 
```