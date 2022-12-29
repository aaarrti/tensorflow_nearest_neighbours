package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "python/ops/_nearest_neighbours_ops.so",
    srcs = [
        "cc/kernels/nearest_neighbours.h",
        "cc/kernels/nearest_neighbours_kernel.cc",
        "cc/ops/nearest_neighbours_op.cc",
    ],
    copts = select({
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-fPIC",
            "-O3",
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
