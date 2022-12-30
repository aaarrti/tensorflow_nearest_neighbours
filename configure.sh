PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
PIP="pip3"

function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

function write_to_bazelrc() {
  echo "$1" >>.bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc


PROJECT_DIR=$(pwd)
TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS="$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
HEADER_DIR=${TF_CFLAGS:2}
SHARED_LIBRARY_NAME=$(echo "$TF_LFLAGS" | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if is_macos; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  elif is_linux; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi

while [[ "$TF_NEED_CUDA" == "" ]]; do
  # shellcheck disable=SC2162
  read -p "Do you want to build op with cuda acceleration. [Y/n] " INPUT
  case $INPUT in
  [Yy]*)
    echo "Build with cuda."
    TF_NEED_CUDA=1
    ;;
  [Nn]*)
    echo "Build without cuda."
    TF_NEED_CUDA=0
    ;;
  "")
    echo "Build without cuda."
    TF_NEED_CUDA=0
    ;;
  *) echo "Invalid selection: " "$INPUT" ;;
  esac
done

while [[ "$TF_NEED_METAL" == "" ]]; do
  # shellcheck disable=SC2162
  read -p "Do you want to build op with metal acceleration. [Y/n] " INPUT
  case $INPUT in
  [Yy]*)
    echo "Build with metal."
    TF_NEED_METAL=1
    ;;
  [Nn]*)
    echo "Build without metal."
    TF_NEED_METAL=0
    ;;
  "")
    echo "Build without metal."
    TF_NEED_METAL=0
    ;;
  *) echo "Invalid selection: " "$INPUT" ;;
  esac
done

write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_HEADER_DIR" ${HEADER_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" ${TF_NEED_METAL}
write_action_env_to_bazelrc "TF_NEED_METAL" ${TF_NEED_METAL}
write_action_env_to_bazelrc "PROJECT_DIR" ${PROJECT_DIR}

if [[ "$TF_NEED_CUDA" == "1" ]]; then
  write_to_bazelrc "build:cuda --define=using_cuda=true --define=using_cuda_nvcc=true"
  write_to_bazelrc "build --config=cuda"
fi



