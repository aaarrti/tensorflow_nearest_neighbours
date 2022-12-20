#!/bin/zsh

PROJECT_DIR=$(realpath "$(pwd)"/..)
TARGET_DIR="/tf_nearest_neighbours"

docker build -t devcontainer .
docker run --name devcontainer -p 2222:22 -d --mount type=bind,source="$PROJECT_DIR",target="$TARGET_DIR" devcontainer
docker exec -it devcontainer /bin/bash