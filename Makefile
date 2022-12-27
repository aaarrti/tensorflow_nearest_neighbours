clean:
	rm -rf build/*

test:
	cp python/nearest_neighbours_test.py build
	cd build; python -m unittest nearest_neighbours_test.TestSO

test_gpu:
	cp python/nearest_neighbours_test.py build;
	cd build; python -m unittest nearest_neighbours_test.TestOnGPU

pip_pkg:
	mkdir build/nearest_neighbours | true
	cp setup.py build/nearest_neighbours
	cp python/__init__.py build/nearest_neighbours
	cp python/nearest_neighbours.py build/nearest_neighbours
	cp MANIFEST.in build/MANIFEST.in
	cp build/*.so build/nearest_neighbours
	cp build/*.metallib build/nearest_neighbours | true
	cd build; python3 nearest_neighbours/setup.py bdist_wheel
	mkdir artifacts | true
	cp build/dist/*.whl artifacts/

metal_lib:
	xcrun -sdk macosx metal -c cc/kernels/nearest_neighbours.metal -o build/_nearest_neighbours.air -ffast-math
	xcrun -sdk macosx metallib build/_nearest_neighbours.air -o build/_nearest_neighbours.metallib


TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
C_FLAGS = ${TF_CFLAGS} -fPIC -std=c++17 -O3
L_FLAGS = -shared ${TF_LFLAGS}


CPU_SRC = cc/ops/nearest_neighbours_op.cc cc/kernels/nearest_neighbours_kernel.cc
CUDA_LIB = build/_nearest_neighbours_kernel.cu.o
TARGET_FLAG = -o build/_nearest_neighbours_op.so

cpu_kernel:
	clang++ $(C_FLAGS) $(L_FLAGS) $(CPU_SRC) $(TARGET_FLAG)

cuda_lib:
	nvcc -I/cc/include -std=c++17 -c $(TF_CFLAGS) $(L_FLAGS) -D CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr cc/kernels/nearest_neighbours_kernel.cu -o $(CUDA_LIB)

cuda_kernel:
	clang++ $(CPU_SRC) $(CUDA_LIB) $(C_FLAGS) $(L_FLAGS) -D CUDA=1 -I/cc/include -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart $(TARGET_FLAG)

metal_kernel:
	clang++ -x objective-c++ $(C_FLAGS) $(L_FLAGS) cc/ops/nearest_neighbours_op.cc cc/kernels/nearest_neighbours_kernel.mm.cc $(TARGET_FLAG) -framework Foundation -undefined dynamic_lookup