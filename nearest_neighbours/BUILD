package(default_visibility = ["//visibility:public"])

load("//metal:metal.bzl", "if_metal_is_enabled")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts =
        [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

# CPU
cc_binary(
    name = "python/ops/_nearest_neighbours_ops.so",
    srcs = [
        "cc/kernels/nearest_neighbours.h",
        "cc/kernels/nearest_neighbours_kernel.cc",
        "cc/ops/nearest_neighbours_op.cc",
    ] + if_metal_is_enabled(["cc/kernels/nearest_neighbours_kernel_metal.cc"]),
    copts = [
        "-pthread",
        "-std=c++17",
        "-fPIC",
        "-O3",
    ] + if_metal_is_enabled([
        "-x objective-c++",
        "-framework Foundation",
        "-undefined dynamic_lookup",
    ]),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_metal_is_enabled([
        ":_nearest_neighbours.metallib",
    ]) + if_cuda_is_configured([
        ":_nearest_neighbours.cu.o",
    ]),
)

cc_library(
    name = "_nearest_neighbours.cu.o",
    srcs = [
        "cc/kernels/nearest_neighbours_kernel.cu",
    ],
    copts = [
        "-pthread",
        "-std=c++17",
        "-fPIC",
        "-O3",
    ] + if_cuda_is_configured([
        "-DTENSORFLOW_USE_NVCC=1",
        "-D CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ]),
    linkopts = [],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        ":cuda",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    alwayslink = 1,
)

genrule(
    name = "_nearest_neighbours.metallib",
    srcs = ["cc/kernels/nearest_neighbours.metal"],
    outs = ["cc/kernels/_nearest_neighbours.metallib"],
    cmd =
        """
    xcrun -sdk macosx metal -c $(locations cc/kernels/nearest_neighbours.metal) -o _nearest_neighbours.air -ffast-math
    xcrun -sdk macosx metallib _nearest_neighbours.air -o $(locations cc/kernels/_nearest_neighbours.metallib)
    """,
)

py_library(
    name = "nearest_neighbours_ops_py",
    srcs = [
        "python/ops/nearest_neighbours_ops.py",
    ],
    data = [
        ":python/ops/_nearest_neighbours_ops.so",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "nearest_neighbours_py_test",
    srcs = [
        "python/ops/nearest_neighbours_test.py",
    ],
    main = "python/ops/nearest_neighbours_test.py",
    srcs_version = "PY3",
    deps = [
        ":nearest_neighbours_ops_py",
    ],
)

py_library(
    name = "nearest_neighbours_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":nearest_neighbours_ops_py",
    ],
)
