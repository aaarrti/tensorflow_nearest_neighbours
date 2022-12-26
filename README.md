# TensorFlow Nearest Neighbours Op


| Tool       | Ubuntu Version | MacOS Version       |
|------------|----------------|---------------------|
| Clang      | 10.0.0         | 14.0.0              |
| Tensorflow | 2.11.0         | 2.11.0              |
| Python     | 3.10           | 3.10                |
| cuda       | 11.2           | -                   | 
| nvcc       | V11.2.152      | -                   | 
| metal      | -              | 31001.667           | 
| metallib   | -              | 31001.667           |                                             

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
    make cuda_lib
    make cuda_kernel
    ```
  - Metal (macOS only)
    ```bash
    make metal_lib
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

#### (https://www.jetbrains.com/help/fleet/generating-a-json-compilation-database.html)[Generate a compilation database for your IDE].