#!/bin/sh

# Filter annoying tf logging messages ...
export TF_CPP_MIN_LOG_LEVEL=2

# Add our multimodel project to the pythonpath
PYTOHNPATH=``pwd`` python 'multimodel/run.py' $@
