set -e

TF_NEED_CUDA_LINE=$(grep -n "TF_NEED_CUDA" .bazelrc)
TF_NEED_CUDA=$(echo "$TF_NEED_CUDA_LINE" | cut -d "=" -f 2)
TF_NEED_METAL_LINE=$(grep -n "TF_NEED_METAL" .bazelrc)
TF_NEED_METAL=$(echo "$TF_NEED_METAL_LINE" | cut -d "=" -f 2)

PIP_FILE_PREFIX="bazel-bin/build_pip_pkg.runfiles/__main__/"
DEST="artifacts"

mkdir -p ${DEST}
DEST=$(pwd -P)/${DEST}

echo "=== destination directory: ${DEST}"
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
echo "$(date)" : "=== Using tmpdir: ${TMPDIR}"
echo "=== Copy TensorFlow Custom op files"

cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
cp ${PIP_FILE_PREFIX}MANIFEST.in "${TMPDIR}"
cp ${PIP_FILE_PREFIX}LICENSE "${TMPDIR}"
rsync -avm -L --exclude='*_test.py' ${PIP_FILE_PREFIX}nearest_neighbours "${TMPDIR}"

pushd "${TMPDIR}"
echo "$(date)" : "=== Building wheel"

if [[ "$TF_NEED_CUDA" == "1" ]]; then
  python3 setup.py egg_info --tag-build=.cuda bdist_wheel
elif [[ "$TF_NEED_METAL" == "1" ]]; then
  python3 setup.py egg_info --tag-build=.metal bdist_wheel
else
  python3 setup.py egg_info --tag-build=.cpu bdist_wheel
fi

cp dist/*.whl "${DEST}"
popd
rm -rf "${TMPDIR}"
echo "$(date)" : "=== Output wheel file is in: ${DEST}"
