#!/bin/bash

# Check that all required arguments are present
if [[ ! $# -eq 1 ]] ; then
    echo 'ERROR: Missing mandatory arguments!'
    echo '       (./run_web_inference.sh <port>)'
    exit 2
fi

PORT=$1

while :; do
  nvidia-docker run -it -p $PORT:9001 -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 \
                     weilemar-vongrdir-ba-ml-fs17 python /BA-ML-FS17/source/web/app.py
  sleep 1
done
