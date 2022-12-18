licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

cc_binary(
    name = "python/ops/_nearest_neighbours_ops.so",
    srcs = [
        "cc/kernels/nearest_neighbours.h",
        "cc/kernels/nearest_neighbours_kernels.cc",
        "cc/ops/nearest_neighbours_ops.cc",
    ],
    copts = select({
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX17_ABI=0",
        ],
    }),
    features = select({
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
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
    name = "nearest_neighbours_ops_py_test",
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


cc_library(
    name = "nearest_neighbours_ops_gpu",
    srcs = [
        "cc/kernels/nearest_neighbours.h",
        "cc/kernels/nearest_neighbours_kernels.cu",
    ],
    copts = select({
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX17_ABI=0",
        ],
    }) + if_cuda_is_configured([
        "-DTENSORFLOW_USE_NVCC=1",
        "-DGOOGLE_CUDA=1",
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
