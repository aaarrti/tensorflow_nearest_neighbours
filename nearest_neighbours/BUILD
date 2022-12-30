package(default_visibility = ["//visibility:public"])

load("//metal:metal.bzl", "if_metal_is_enabled")
load("//cuda:cuda.bzl", "if_cuda_is_enabled")

# CPU
cc_binary(
    name = "python/ops/_nearest_neighbours_ops.so",
    srcs = [
        "cc/kernels/nearest_neighbours.h",
        "cc/kernels/nearest_neighbours_kernel.cc",
        "cc/ops/nearest_neighbours_op.cc",
    ] + if_metal_is_enabled(["cc/kernels/nearest_neighbours_kernel_metal.cc"]),
    copts = select({
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-fPIC",
            "-O3",
        ] + if_metal_is_enabled([
            "-x",
            "objective-c++",
            "-framework ",
            "Foundation",
            "-undefined",
            "dynamic_lookup",
        ]),
    }),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_metal_is_enabled([
        ":_nearest_neighbours.metallib",
    ]) + if_cuda_is_enabled([
        ":_nearest_neighbours.cu.o",
    ]),
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

#cc_library(
#    name = "_nearest_neighbours.cu.o",
#)
