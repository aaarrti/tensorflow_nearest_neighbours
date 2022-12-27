# TensorFlow Nearest Neighbours Op


| Tool       | Ubuntu    | MacOS     |
|------------|-----------|-----------|
| OS         | 20.04.5   |   12.6.1  |
| Clang      | 10.0.0    | 14.0.0    |
| Tensorflow | 2.11.0    | 2.11.0    | 
| Python     | 3.9       | 3.8       |
| cuda       | 11.2      | -         | 
| nvcc       | V11.2.152 | -         | 
| metal      | -         | 31001.667 | 
| metallib   | -         | 31001.667 |                                             

### Building from source:
- First we need to build the shared object (library)
  - CPU only:
    ```bash
    make cpu_kernel
    ```
  - CUDA (linux only)
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
make test_so
```
- Afterwards, we build a pip package from it:
  - CPU version
  ```bash
  make pip_pkg_cpu
  ```
  - Cuda version
  ```bash
  make pip_pkg_cuda
  ```
  - Metal version
  ```bash
  make pip_pkg_metal
  ```
  

- And finally, we can install it: 
```bash
pip install build/dist/*.whl 
```
- Verify it works
```python
import tensorflow as tf
from nearest_neighbours import nearest_neighbours
tf.debugging.set_log_device_placement(True)

x = tf.random.uniform(shape=[8, 10, 32])
em = tf.random.uniform(shape=[500, 32])
result = nearest_neighbours(x, em)
print(result.shape)
```