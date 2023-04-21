FROM --platform=$BUILDPLATFORM tensorflow/tensorflow:2.10.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
# Install make, clang
RUN apt update && apt install -y make clang
# Install python3.{8, 9, 10}
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 python3.9 python3.10 \
RUN pip install tox

WORKDIR /tensorflow_nearest_neighbours