#!/bin/zsh

set -e

PROJECT_DIR=$(realpath "$(pwd)"/..)
docker run --mount type=bind,source="$PROJECT_DIR",target="/tensorflow_nearest_neighbours" \
  --platform=linux/x86_64 --rm -it devcontainer "$1"