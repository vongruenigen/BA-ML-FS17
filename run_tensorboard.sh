#!/bin/bash
#
# This script enables to run tensorboard in docker. It uses the first
# parameter as the port which tensorboard should be mapped to and the
# second as the logdir.
#

# Check that all required arguments are present
if [[ ! $# -eq 2 ]] ; then
    echo 'ERROR: Missing mandatory arguments!'
    echo '       (./run_tensorboard.sh <port> <logdir>)'
    exit 2
fi

PORT=$1
LOGDIR=$2

while :; do
  nvidia-docker run -it -p $PORT:8008 -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 \
                     weilemar-vongrdir-ba-ml-fs17 python -m tensorflow.tensorboard --port=$PORT \
                                                         --logdir=$LOGDIR
  sleep 1
done
