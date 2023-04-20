#!/bin/zsh

set -e

docker build -t devcontainer --platform=linux/x86_64 .

PROJECT_DIR=$(realpath "$(pwd)"/..)
docker run --mount type=bind,source="$PROJECT_DIR",target="/tensorflow_nearest_neighbours" \
  --name devcontainer --platform=linux/x86_64 --rm -it devcontainer