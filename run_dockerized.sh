#!/bin/bash
docker build -t weilemar-vongrdir-ba-ml-fs17 .
nvidia-docker run -it -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 \
                  --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl \
                  --device /dev/nvidia-uvm:/dev/nvidia-uvmweilemar-vongrdir-ba-ml-fs17 /BA-ML-FS17/run_seq2seq.sh
