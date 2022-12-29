set -e
set -x

TF_NEED_CUDA_LINE=$(grep -n "TF_NEED_CUDA" .bazelrc)
TF_NEED_CUDA=$(echo "$TF_NEED_CUDA_LINE" | cut -d "=" -f 2)
TF_NEED_METAL_LINE=$(grep -n "TF_NEED_METAL" .bazelrc)
TF_NEED_METAL=$(echo "$TF_NEED_METAL_LINE" | cut -d "=" -f 2)
PROJECT_DIR_LINE=$(grep -n "PROJECT_DIR" .bazelrc)
PROJECT_DIR=$(echo "$PROJECT_DIR_LINE" | cut -d "=" -f 2)
DEST="artifacts"

mkdir -p ${DEST}
DEST=$(pwd -P)/${DEST}

echo "=== destination directory: ${DEST}"
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
echo "$(date)" : "=== Using tmpdir: ${TMPDIR}"
echo "=== Copy TensorFlow Custom op files"

cp setup.py "${TMPDIR}"
cp MANIFEST.in "${TMPDIR}"
cp LICENSE "${TMPDIR}"
rsync -avm -L --exclude='*_test.py' nearest_neighbours "${TMPDIR}"

pushd "${TMPDIR}"
echo "$(date)" : "=== Building wheel"

if [[ "$TF_NEED_CUDA" == "1" ]]; then
  python3 setup.py egg_info --tag-build=.cuda bdist_wheel
elif [[ "$TF_NEED_METAL" == "1" ]]; then
  python3 setup.py egg_info --tag-build=.metal bdist_wheel
else
  python3 setup.py egg_info --tag-build=.cpu bdist_wheel
fi

PROJECT_ARTIFACTS_DIR="${PROJECT_DIR}/artifacts"
PROJECT_ARTIFACTS_DIR="$(echo "${PROJECT_ARTIFACTS_DIR}" | tr -d '"')"
cp dist/*.whl "$PROJECT_ARTIFACTS_DIR"
popd
rm -rf "${TMPDIR}"
echo "$(date)" : "=== Output wheel file is in: ${PROJECT_ARTIFACTS_DIR}"
