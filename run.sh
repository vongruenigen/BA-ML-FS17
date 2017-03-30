#!/bin/sh

# Only show TF log messages if DEBUG is set
if [ ! -z "$DEBUG" ]; then
  export TF_CPP_MIN_LOG_LEVEL=2
fi

python 'source/run.py' $@
