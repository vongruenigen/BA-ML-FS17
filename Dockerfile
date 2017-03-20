FROM gcr.io/tensorflow/tensorflow:latest-py3

MAINTAINER MARTIN WEILENMANN <weilemar@students.zhaw.ch>

# Clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /BA-ML-FS17/
RUN pip install cython
RUN pip install -r requirements.txt

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888

RUN ["/bin/bash"]
