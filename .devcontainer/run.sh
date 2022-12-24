#!/bin/zsh

PROJECT_DIR=$(realpath "$(pwd)"/..)

docker build -t devcontainer .
docker run --name devcontainer --mount type=bind,source="$PROJECT_DIR",target="/devcontainer/tf_nearest_neighbours" --rm -it devcontainer