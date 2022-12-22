test:
	cp python/nearest_neighbours_test.py build;
	cd build; python nearest_neighbours_test.py

pip_pkg:
	cp -f setup.py build
	cp -f python/__init__.py build
	cp -f python/nearest_neighbours.py build
	cd build; python setup.py bdist_wheel


TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


CPU_SRC = cc/ops/nearest_neighbours_ops.cc cc/kernels/nearest_neighbours_kernels.cc
CUDA_LIB = build/_nearest_neighbours_ops.cu.o

C_FLAGS = ${TF_CFLAGS} -fPIC -std=c++17 -O3
L_FLAGS = -shared ${TF_LFLAGS}
TARGET_FLAG = -o build/_nearest_neighbours_ops.so

cpu_kernel:
	g++ $(C_FLAGS) $(L_FLAGS) $(CPU_SRC) $(TARGET_FLAG)

cuda_kernel:
	nvcc -c $(TF_CFLAGS) $(L_FLAGS) \
		-D CUDA=1 \
		-x cu -Xcompiler -fPIC --expt-relaxed-constexpr \
		cc/kernels/nearest_neighbours_kernels.cu \
		-o $(CUDA_LIB)
	g++ $(C_FLAGS) $(L_FLAGS) \
		-D CUDA=1  \
		-I/usr/local/cuda/targets/x86_64-linux/include \
		-L/usr/local/cuda/targets/x86_64-linux/lib -lcudart \
		$(CPU_SRC) $(CUDA_LIB) $(TARGET_FLAG)


metal_kernel:
	xcrun -sdk macosx metal \
		-c cc/kernels/nearest_neighbours_kernels.metal \
		-o build/_nearest_neighbours_kernels.air \
		-ffast-math
	xcrun -sdk macosx metallib \
		build/_nearest_neighbours_kernels.air \
		-o build/_nearest_neighbours_kernels.metallib
	clang++ -x objective-c++ \
		$(C_FLAGS) $(L_FLAGS) \
		cc/kernels/nearest_neighbours_kernels.cc \
		cc/ops/nearest_neighbours_ops.cc \
		cc/kernels/nearest_neighbours_kernels.mm.cc $(TARGET_FLAG) \
		-framework Foundation -undefined dynamic_lookup


clean:
	rm -r -f build/*

