#!/bin/sh

set -e

docker run \
  --mount type=bind,source="$(realpath "$(pwd)")",target="/tensorflow_nearest_neighbours" \
  --platform=linux/x86_64 --rm devcontainer "$@"
