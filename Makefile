TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++17
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = build/_nearest_neighbours_ops.so
CPU_SRC = cc/ops/nearest_neighbours_ops.cc cc/kernels/nearest_neighbours_kernels.cc

test:
	cd tests; python nearest_neighbours_test.py

pip_pkg:
	cp -f setup.py build
	cp -f python/__init__.py build
	cp -f python/nearest_neighbours.py build
	cd build; python3 setup.py bdist_wheel

cpu_kernel: clean
	mkdir build
	g++ $(CFLAGS) $(LDFLAGS) $(CPU_SRC) -o $(TARGET_LIB) -O3

cuda_kernel: clean
	mkdir build
	nvcc -std=c++11 $(CFLAGS) $(LDFLAGS) -D CUDA=1 -shared $(CPU_SRC) cuda/nearest_neighbours_kernels.cu -o $(TARGET_LIB) -O3

metal_kernel: clean
	mkdir build
	xcrun -sdk macosx metal -c metal/nearest_neighbours_kernels.metal -o build/nearest_neighbours_kernels.air -ffast-math
	xcrun -sdk macosx metallib build/nearest_neighbours_kernels.air -o build/nearest_neighbours_kernels.metallib

	clang++ -x objective-c++ -std=c++14 -shared cc/ops/nearest_neighbours_ops.cc -shared cc/kernels/nearest_neighbours_kernels.cc \
    metal/metal_nearest_neighbours_kernels.cc -o build/_nearest_neighbours_ops.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O3 -framework Foundation -undefined dynamic_lookup


clean:
	rm -r -f build
