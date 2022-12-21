FROM tensorflow/tensorflow:devel-gpu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /devcontainer

RUN apt update && apt install software-properties-common clang && add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 && python3.9 -m pip install tensorflow==2.11.0