#!/bin/bash
nvidia-docker run -it \
                  -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 \
                  weilemar-vongrdir-ba-ml-fs17 /BA-ML-FS17/run.sh "$@"
