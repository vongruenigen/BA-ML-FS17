FROM gcr.io/tensorflow/tensorflow:latest-gpu-py3

LABEL authors="Dirk von Grünigen, Martin Weilenmann"

# Clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y libtcmalloc-minimal4

WORKDIR /BA-ML-FS17/
ADD /requirements.txt /BA-ML-FS17/requirements.txt
RUN pip install cython
RUN pip install -r /BA-ML-FS17/requirements.txt
RUN python -m nltk.downloader punkt

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888

RUN ["/bin/bash"]
