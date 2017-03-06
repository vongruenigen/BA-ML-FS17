#!/bin/sh

# Filter annoying tf logging messages ...
export TF_CPP_MIN_LOG_LEVEL=2

python 'source/run.py' $@
