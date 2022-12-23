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


CPU_SRC = cc/ops/nearest_neighbours_op.cc cc/kernels/nearest_neighbours_kernel.cc
CUDA_LIB = build/_nearest_neighbours_op.cu.o

C_FLAGS = ${TF_CFLAGS} -fPIC -std=c++17 -O3
L_FLAGS = -shared ${TF_LFLAGS}
TARGET_FLAG = -o build/_nearest_neighbours_op.so

cpu_kernel:
	g++ $(C_FLAGS) $(L_FLAGS) $(CPU_SRC) $(TARGET_FLAG)

cuda_lib:
	nvcc -c $(TF_CFLAGS) $(L_FLAGS) \
		-D CUDA=1 \
		-x cu -Xcompiler -fPIC --expt-relaxed-constexpr \
		cc/kernels/nearest_neighbours_kernel.cu \
		-o $(CUDA_LIB)


cuda_kernel:
	g++ $(C_FLAGS) $(L_FLAGS) \
		-D CUDA=1  \
		-I/usr/local/cuda/targets/x86_64-linux/include \
		-L/usr/local/cuda/targets/x86_64-linux/lib -lcudart \
		$(CPU_SRC) $(CUDA_LIB) $(TARGET_FLAG)

metal_lib:
	xcrun -sdk macosx metal \
		-c cc/kernels/nearest_neighbours_kernel.metal \
		-o build/_nearest_neighbours_kernel.air \
		-ffast-math
	xcrun -sdk macosx metallib \
		build/_nearest_neighbours_kernel.air \
		-o build/_nearest_neighbours_kernel.metallib

metal_kernel:
	clang++ -x objective-c++ \
		$(C_FLAGS) $(L_FLAGS) \
		cc/kernels/nearest_neighbours_kernel.cc \
		cc/ops/nearest_neighbours_op.cc \
		cc/kernels/nearest_neighbours_kernel.metal.cc $(TARGET_FLAG) \
		-framework Foundation -undefined dynamic_lookup

clean:
	rm -r -f build/*

