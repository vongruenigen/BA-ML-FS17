#!/bin/bash
nvidia-docker run -it \
                  -v `pwd`:/BA-ML-FS17 \
                  weilemar-vongrdir-ba-ml-fs17 /BA-ML-FS17/run.sh "$@"
