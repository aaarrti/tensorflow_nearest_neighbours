.PHONY: test

clean:
	rm -rf tensorflow_nearest_neighbours/*.so

test: pip_pkg
	python -m pip install dist/*.whl --force-reinstall --no-deps
	python -m unittest test/nearest_neighbours_test.py

pip_pkg:
	python3 setup.py bdist_wheel


TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
C_FLAGS = ${TF_CFLAGS} -fPIC -std=c++17 -O3
L_FLAGS = -shared ${TF_LFLAGS}

CPU_SRC = tensorflow_nearest_neighbours/cc/ops/nearest_neighbours_op.cc tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.cc
CUDA_LIB = tensorflow_nearest_neighbours/_nearest_neighbours_kernel.cu.o
METAL_LIB = tensorflow_nearest_neighbours/_nearest_neighbours_kernel.metal.o
TARGET_FLAG = -o tensorflow_nearest_neighbours/_nearest_neighbours_op.so

cpu_kernel:
	clang++ $(C_FLAGS) $(L_FLAGS) $(CPU_SRC) $(TARGET_FLAG)

cuda_kernel:
	nvcc -I/cc/include -std=c++17 -c $(TF_CFLAGS) $(L_FLAGS) -D CUDA=1 -x cu -Xcompiler -o $(CUDA_LIB) \
    		--expt-relaxed-constexpr tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.cu \
	clang++ $(CPU_SRC) $(CUDA_LIB) $(C_FLAGS) $(L_FLAGS) -D CUDA=1 -I/cc/include \
		-I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart $(TARGET_FLAG)

metal_kernel:
	xcrun -sdk macosx metal -c tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours.metal \
		-o tensorflow_nearest_neighbours/_nearest_neighbours.air -ffast-math
	xcrun -sdk macosx metallib tensorflow_nearest_neighbours/_nearest_neighbours.air \
		-o tensorflow_nearest_neighbours/_nearest_neighbours.metallib
	clang++ -x objective-c++ -framework Foundation -undefined dynamic_lookup $(C_FLAGS) $(L_FLAGS) \
	tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.mm.cc -o $(METAL_LIB)
	clang++ $(CPU_SRC) $(METAL_LIB) $(C_FLAGS) $(L_FLAGS) $(TARGET_FLAG)