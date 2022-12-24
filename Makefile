clean:
	rm -r -f build/*

test:
	cp python/nearest_neighbours_test.py build;
	cd build; python nearest_neighbours_test.py

pip_pkg:
	cp -f setup.py build
	cp -f python/__init__.py build
	cp -f python/nearest_neighbours.py build
	cd build; python setup.py bdist_wheel

metal_lib:
	xcrun -sdk macosx metal -c cc/kernels/nearest_neighbours_kernel.metal -o build/_nearest_neighbours_kernel.air -ffast-math
	xcrun -sdk macosx metallib build/_nearest_neighbours_kernel.air -o build/_nearest_neighbours_kernel.metallib


TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
#TF_CFLAGS=-I/Users/artemsereda/anaconda3/envs/custom_op/lib/python3.9/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
#TF_LFLAGS=-L/Users/artemsereda/anaconda3/envs/custom_op/lib/python3.9/site-packages/tensorflow -ltensorflow_framework.2

CPU_SRC = cc/ops/nearest_neighbours_op.cc cc/kernels/nearest_neighbours_kernel.cc
CUDA_LIB = build/_nearest_neighbours_kernel.cu.o

C_FLAGS = ${TF_CFLAGS} -fPIC -std=c++17 -O3
L_FLAGS = -shared ${TF_LFLAGS}
TARGET_FLAG = -o build/_nearest_neighbours_op.so

cpu_kernel:
	g++ $(C_FLAGS) $(L_FLAGS) $(CPU_SRC) $(TARGET_FLAG)

cuda_lib:
	nvcc -std=c++17 -c $(TF_CFLAGS) $(L_FLAGS) -D CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr cc/kernels/nearest_neighbours_kernel.cu -o $(CUDA_LIB)

cuda_kernel:
	g++ $(CPU_SRC) $(CUDA_LIB) $(C_FLAGS) $(L_FLAGS) -D CUDA=1 -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart $(TARGET_FLAG)

metal_kernel:
	clang++ -Icc/include -D METAL=1 $(C_FLAGS) $(L_FLAGS) $(CPU_SRC) cc/kernels/nearest_neighbours_kernel.metal.cc $(TARGET_FLAG) -framework Foundation -framework QuartzCore -framework Metal