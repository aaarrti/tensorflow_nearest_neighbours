# This is a Makefile to compile the custom op to a .so file.

TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


cpu_kernel:
	mkdir build | true
	g++ cc/

cuda_kernel:
	mkdir build | true

metal_kernel:
	mkdir build | true
	xcrun -sdk macosx metal -c nearest_neighbours/metal/nearest_neighbours_kernels.metal -o build/nearest_neighbours_kernels.air -ffast-math
	xcrun -sdk macosx metallib build/nearest_neighbours_kernels.air -o build/nearest_neighbours_kernels.metallib

	clang++ -x objective-c++ -std=c++14 -shared nearest_neighbours/cc/ops/nearest_neighbours_ops.cc \
    nearest_neighbours/metal/metal_nearest_neighbours_kernels.cc \
	-o build/_nearest_neighbours_ops.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O3 -framework Foundation -undefined dynamic_lookup


clean:
	rm -r -f build
