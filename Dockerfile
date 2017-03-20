FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

MAINTAINER MARTIN WEILENMANN <weilemar@students.zhaw.ch>

# Clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y git

WORKDIR /BA-ML-FS17/

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888

RUN ["/bin/bash"]
