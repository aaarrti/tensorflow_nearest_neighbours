# This is a Makefile to compile the custom op to a .so file.

TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
#-I/Users/artemsereda/anaconda3/envs/custom_op/lib/python3.9/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 --std=c++17 -DEIGEN_MAX_ALIGN_BYTES=64
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
# -L/Users/artemsereda/anaconda3/envs/custom_op/lib/python3.9/site-packages/tensorflow -ltensorflow_framework.2

kernel:
	xcrun -sdk macosx metal \
		-c nearest_neighbours/cc/kernels/nearest_neighbours_kernels.metal \
		-o nearest_neighbours/cc/kernels/nearest_neighbours_kernels.air \
		-ffast-math
	xcrun -sdk macosx metallib \
		nearest_neighbours/cc/kernels/nearest_neighbours_kernels.air \
		-o nearest_neighbours/cc/kernels/nearest_neighbours_kernels.metallib
	clang++ -x objective-c++ -std=c++14 \
		-shared nearest_neighbours/cc/kernels/nearest_neighbours_kernels.cc \
		-shared nearest_neighbours/cc/ops/nearest_neighbours_ops.cc \
		nearest_neighbours/cc/kernels/mtl_nearest_neighbours_kernels.cc \
		-o nearest_neighbours/cc/kernels/nearest_neighbours_kernels.so \
		-fPIC $(TF_CFLAGS) $(TF_LFLAGS) \
		-O3 -framework Foundation -undefined dynamic_lookup

clean:
	rm -f nearest_neighbours/cc/kernels/nearest_neighbours_kernels.so
	rm -f nearest_neighbours/cc/kernels/nearest_neighbours_kernels.air
	rm -f nearest_neighbours/cc/kernels/nearest_neighbours_kernels.metallib
	
