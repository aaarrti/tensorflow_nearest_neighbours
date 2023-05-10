cmake_minimum_required(VERSION 3.16.3)
set(CMAKE_VERBOSE_MAKEFILE ON)

if (${EnableMetal})
    project(tensorflow_nearest_neighbours LANGUAGES CXX OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 11)
    include_directories(/Users/artemsereda/miniconda3/envs/tensorflow_nearest_neighbours/lib/python3.9/site-packages/tensorflow/include)
    link_directories(/Users/artemsereda/miniconda3/envs/tensorflow_nearest_neighbours/lib/python3.9/site-packages/tensorflow)
elseif (${EnableCUDA})
    project(tensorflow_nearest_neighbours CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    include_directories(/usr/local/lib/python3.8/dist-packages/tensorflow/include)
    link_directories(/usr/local/lib/python3.8/dist-packages/tensorflow)
else ()
    project(tensorflow_nearest_neighbours LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 11)
    SET(CMAKE_CXX_COMPILER "/usr/bin/clang++")
    project(tensorflow_nearest_neighbours)
    include_directories(/Users/artemsereda/miniconda3/envs/tensorflow_nearest_neighbours/lib/python3.9/site-packages/tensorflow/include)
    link_directories(/Users/artemsereda/miniconda3/envs/tensorflow_nearest_neighbours/lib/python3.9/site-packages/tensorflow)
endif()



set(CMAKE_SHARED_LIBRARY_PREFIX "")
add_compile_options(-O3 "-ferror-limit=0")
option(EnableMetal "EnableMetal" OFF)
option(EnableMetal "EnableCUDA" OFF)


if (${EnableMetal})
    add_custom_command(
            OUTPUT "${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.air"
            COMMAND xcrun -sdk macosx metal -ffast-math -c ${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours.metal -o ${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.air
    )
    add_custom_command(
            OUTPUT "${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.metallib"
            COMMAND xcrun -sdk macosx metallib ${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.air -o ${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.metallib
            DEPENDS "${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.air"
    )
    add_custom_target(
            metal_lib
            DEPENDS "${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/_nearest_neighbours.metallib"

    )
    set(CMAKE_EXE_LINKER_FLAGS "-undefined dynamic_lookup -framework Foundation -framework Metal")
    add_library(
            metal_kernel
            tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.mm
    )
elseif (${EnableCUDA})
    set(CMAKE_CXX_FLAGS "-D CUDA=1")
    add_library(
            cuda_lib
            SHARED
            tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.cu
    )
    set_target_properties(
            cuda_lib
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION
            ON
    )
    target_link_libraries(cuda_lib tensorflow_framework.2)
else()
    add_library(
            _nearest_neighbours
            SHARED
            tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.cc
            tensorflow_nearest_neighbours/cc/ops/nearest_neighbours_op.cc
    )
    target_link_libraries(_nearest_neighbours tensorflow_framework.2)
endif ()


