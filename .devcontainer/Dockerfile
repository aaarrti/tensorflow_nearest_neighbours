FROM tensorflow/tensorflow:devel-gpu

ENV DEBIAN_FRONTEND=noninteractive

COPY cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz .

RUN pip install tensorflow==2.11.0 && \
    apt-get install zlib1g && \
    tar -xvf cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz \
    cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include && \
    cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*