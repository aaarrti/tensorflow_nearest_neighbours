load(":build_def.bzl", "metallib")
package(default_visibility = ["//visibility:public"])

metallib(
    name = "_nearest_neighbours.metallib",
    input = "cc/kernels/nearest_neighbours.metal",
    output = "python/ops/_nearest_neighbours.metallib",
)

