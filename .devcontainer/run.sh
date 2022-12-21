#!/bin/zsh

PROJECT_DIR=$(realpath "$(pwd)"/..)
TARGET_DIR="/devcontainer/tf_nearest_neighbours"

docker build -t devcontainer .
docker run --name devcontainer --mount type=bind,source="$PROJECT_DIR",target="$TARGET_DIR" --rm -it devcontainer