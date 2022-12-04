#!/usr/bin/env bash
set -e
set -x

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"


PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"

function main() {
  while [[ ! -z "${1}" ]]; do
    DEST=${1}
    shift
  done

  if [[ -z ${DEST} ]]; then
    echo "No destination dir provided"
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  mkdir -p ${DEST}
  if [[ ${PLATFORM} == "darwin" ]]; then
    DEST=$(pwd -P)/${DEST}
  else
    DEST=$(readlink -f "${DEST}")
  fi
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy TensorFlow Custom op files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
  cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
  rsync -avm -L ${PIP_FILE_PREFIX}tensorflow_nearest_neighbours "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  python3 setup.py bdist_wheel

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
