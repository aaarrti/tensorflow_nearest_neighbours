cmake_minimum_required(VERSION 3.25)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(tensorflow_nearest_neighbours)
project(tensorflow_nearest_neighbours LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)
SET(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
set(CMAKE_SHARED_LIBRARY_PREFIX "")
add_compile_options(-O3 "-ferror-limit=0")
option(EnableMetal "EnableMetal" OFF)
option(EnableMetal "EnableCUDA" OFF)

# set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/tensorflow_nearest_neighbours/")

if (DEFINED $ENV{PYTHON_CMD})
    set(PYTHON_CMD $ENV{PYTHON_CMD})
else ()
    find_package(Python COMPONENTS Interpreter)
    set(PYTHON_CMD ${Python_EXECUTABLE})
endif ()
message("PYTHON_CMD:" $ENV{PYTHON_CMD})

include_directories(/Users/artemsereda/miniconda3/envs/tensorflow_nearest_neighbours/lib/python3.9/site-packages/tensorflow/include)
link_directories(/Users/artemsereda/miniconda3/envs/tensorflow_nearest_neighbours/lib/python3.9/site-packages/tensorflow)

#execute_process(
#        COMMAND ${PYTHON_CMD} "-c 'import tensorflow as tf;print(\" \".join(tf.sysconfig.get_compile_flags()))'"
#        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
#        RESULT_VARIABLE TF_COMPILE_FLAGS_RESULT
#        OUTPUT_VARIABLE TF_COMPILE_FLAGS
#        COMMAND_ERROR_IS_FATAL ANY
#        OUTPUT_STRIP_TRAILING_WHITESPACE
#        COMMAND_ECHO STDOUT
#)
#message("TF_COMPILE_FLAGS = ${TF_COMPILE_FLAGS}")

add_library(
        _nearest_neighbours
        SHARED
        tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.cc
        tensorflow_nearest_neighbours/cc/ops/nearest_neighbours_op.cc
)
target_link_libraries(_nearest_neighbours tensorflow_framework.2)

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
    target_link_libraries(
            _nearest_neighbours metal_kernel
    )
elseif (${EnableCUDA})
    set(CMAKE_CUDA_STANDARD 18)
    set(CMAKE_CXX_FLAGS "-D CUDA=1")
    add_library(
            cuda_lib
            tensorflow_nearest_neighbours/cc/kernels/nearest_neighbours_kernel.cu
    )
    set_target_properties(
            cuda_lib
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION
            ON
    )
    target_link_libraries(cuda_lib tensorflow_framework.2)
    target_link_libraries(_nearest_neighbours cudart cuda_kernel)
endif ()


