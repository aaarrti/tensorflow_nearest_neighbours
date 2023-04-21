FROM --platform=$BUILDPLATFORM tensorflow/tensorflow:2.10.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
# Install make, clang
RUN apt update && apt install -y make clang

WORKDIR /tensorflow_nearest_neighbours