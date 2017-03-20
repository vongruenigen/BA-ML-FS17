#!/bin/bash
# docker build -t weilemar-vongrdir-ba-ml-fs17 .
nvidia-docker run -it \
                  -p 8750:8008 \
                  -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 \
                  weilemar-vongrdir-ba-ml-fs17 /BA-ML-FS17/run_seq2seq.sh
