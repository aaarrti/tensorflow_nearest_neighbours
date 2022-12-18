#!/bin/bash

set -e
set -x

function write_to_bazelrc() {
  echo "$1" >>.bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

write_action_env_to_bazelrc "TMP" "/tmp"


# Check if we are building GPU or CPU ops, default CPU
while [[ "$TF_NEED_CUDA" == "" ]]; do
  read -p "Do you want to build ops again TensorFlow CPU pip package?"\
" Y or enter for CPU (tensorflow-cpu), N for GPU (tensorflow). [Y/n] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build with CPU pip package."; TF_NEED_CUDA=0;;
    [Nn]* ) echo "Build with GPU pip package."; TF_NEED_CUDA=1;;
    "" ) echo "Build with CPU pip package."; TF_NEED_CUDA=0;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done

# shellcheck disable=SC2207
TF_CFLAGS=($(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
TF_LFLAGS="$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"

SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
fi

HEADER_DIR=${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}

write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_CUDA}

if [[ "$TF_NEED_CUDA" == "1" ]]; then
  write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
  if nvcc --version 2 &>/dev/null; then
    # Determine CUDA version using default nvcc binary
    CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  elif /usr/local/cuda/bin/nvcc --version 2 &>/dev/null; then
    # Determine CUDA version using /usr/local/cuda/bin/nvcc binary
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
  elif [ -f "/usr/local/cuda/version.txt" ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.* \([0-9]\+\.[0-9]\+\).*/\1/')
  else
    CUDA_VERSION=""
  fi
  write_action_env_to_bazelrc "TF_CUDA_VERSION" "${CUDA_VERSION}"
  # TODO parse version from cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR
  write_action_env_to_bazelrc "TF_CUDNN_VERSION" "8"

  write_action_env_to_bazelrc "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
  write_action_env_to_bazelrc "CUDA_TOOLKIT_PATH" "/usr/local/cuda"
  write_to_bazelrc "build --config=cuda"
  write_to_bazelrc "test --config=cuda"
fi
