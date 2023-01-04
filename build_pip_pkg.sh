set -e
set -x

TF_NEED_CUDA=$(grep -n "TF_NEED_CUDA" .bazelrc)
TF_NEED_CUDA=$(echo "$TF_NEED_CUDA" | cut -d "=" -f 2)
TF_NEED_CUDA="$(echo "${TF_NEED_CUDA}" | tr -d '"')"

TF_NEED_METAL=$(grep -n "TF_NEED_METAL" .bazelrc)
TF_NEED_METAL=$(echo "$TF_NEED_METAL" | cut -d "=" -f 2)
TF_NEED_METAL="$(echo "${TF_NEED_METAL}" | tr -d '"')"

DEST="${BUILD_WORKING_DIRECTORY}/artifacts"
DEST="$(echo "${DEST}" | tr -d '"')"
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

if [[ "${TF_NEED_CUDA}" == "1" ]]; then
  python3 setup.py egg_info --tag-build=.cuda bdist_wheel
elif [[ "${TF_NEED_METAL}" == "1" ]]; then
  python3 setup.py egg_info --tag-build=.metal bdist_wheel
else
  python3 setup.py egg_info --tag-build=.cpu bdist_wheel
fi

cp dist/*.whl "$DEST"
popd
rm -rf "${TMPDIR}"
echo "$(date)" : "=== Output wheel file is in: ${DEST}"
