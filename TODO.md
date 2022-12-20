```
ERROR: /home/artem/tf_nearest_neighbours/nearest_neighbours/BUILD:68:11: in deps attribute of cc_library rule //nearest_neighbours:nearest_neighbours_ops_gpu: target '//nearest_neighbours:cuda' does not exist
ERROR: /home/artem/tf_nearest_neighbours/nearest_neighbours/BUILD:68:11: Analysis of target '//nearest_neighbours:nearest_neighbours_ops_gpu' failed
ERROR: Analysis of target '//nearest_neighbours:nearest_neighbours_ops_gpu' failed; build aborted: 
```

```shell
export PATH="$PATH:$HOME/bin"
```

```shell
/home/artem/.cache/bazel/_bazel_artem/install/616d39adb94ebfda4ae84d4ce81a9faf/linux-sandbox \
  -t 15 \
  -w /home/artem/.cache/bazel/_bazel_artem/101a04c3fc595a2d367e095c4794a354/sandbox/linux-sandbox/5/execroot/__main__ \
  -w /tmp \
  -w /dev/shm \
  -D -- /usr/bin/gcc \
  -U_FORTIFY_SOURCE \
  -fstack-protector \
  -Wall \
  -Wunused-but-set-parameter \
  -Wno-free-nonheap-object \
  -fno-omit-frame-pointer \
  -g0 \
  -O2 '-D_FORTIFY_SOURCE=1' \
  -DNDEBUG \
  -ffunction-sections \
  -fdata-sections \
  '-std=c++0x' \
  -MD \
  -MF bazel-out/k8-opt/bin/nearest_neighbours/_objs/nearest_neighbours_ops_gpu/nearest_neighbours.pic.d \
  '-frandom-seed=bazel-out/k8-opt/bin/nearest_neighbours/_objs/nearest_neighbours_ops_gpu/nearest_neighbours.pic.o' \
  -fPIC \
  -iquote . \
  -iquote bazel-out/k8-opt/bin \
  -iquote external/local_config_tf \
  -iquote bazel-out/k8-opt/bin/external/local_config_tf \
  -iquote external/local_config_cuda \
  -iquote bazel-out/k8-opt/bin/external/local_config_cuda \
  -Ibazel-out/k8-opt/bin/external/local_config_cuda/cuda/_virtual_includes/cuda_headers_virtual \
  -isystem external/local_config_tf/include \
  -isystem bazel-out/k8-opt/bin/external/local_config_tf/include \
  -isystem external/local_config_cuda/cuda \
  -isystem bazel-out/k8-opt/bin/external/local_config_cuda/cuda \
  -isystem external/local_config_cuda/cuda/cuda/include \
  -isystem bazel-out/k8-opt/bin/external/local_config_cuda/cuda/cuda/include \
  -pthread \
  '-std=c++14' \
  '-D_GLIBCXX_USE_CXX11_ABI=0' \
  '-DTENSORFLOW_USE_NVCC=1' \
  '-DGOOGLE_CUDA=1' \
  -fno-canonical-system-headers \
  -Wno-builtin-macro-redefined \
  '-D__DATE__="redacted"' \
  '-D__TIMESTAMP__="redacted"' \
  '-D__TIME__="redacted"' \
  -c nearest_neighbours/cc/kernels/nearest_neighbours.cc \
  -o bazel-out/k8-opt/bin/nearest_neighbours/_objs/nearest_neighbours_ops_gpu/nearest_neighbours.pic.o
```