#!/bin/bash
#
# This script enables to run parameterized experiments using the seq2seq tool
# found at github.com/google/seq2seq. The list of parameters is as follows:
#
#   1. Name of the results directory in results/, crashes if it already exists. If the
#      environment variable FORCE is set, it deletes the directory automatically if it
#      already exists.
#   2. List of comma separated file names of YAML files to load the configurations from.
#      The paths are relative to the configs/seq2seq directory where all the configs lie.
#

# Check that all required arguments are present
if [[ ! $# -eq 2 ]] ; then
    echo 'ERROR: Missing mandatory arguments!'
    echo '       (./run_seq2seq.sh <out-name> <list-of-yml-files>)'
    exit 2
fi

# Convenience function for logging errors
echoerr() {
  echo "$@" 1>&2;
}

SEQ2SEQ_DIR_NAME="seq2seq"

if [ ! -d $SEQ2SEQ_DIR_NAME ]; then
  echo "Cloning google/seq2seq project from github and store it in $SEQ2SEQ_DIR_NAME..."
  git clone https://github.com/google/seq2seq.git $SEQ2SEQ_DIR_NAME
fi

cd $SEQ2SEQ_DIR_NAME

RESULTS_NAME=$1
IN_YML_CONFIGS=$2
YML_CONFIGS=""
S2S_PREFIX='../configs/seq2seq'

IFS=',' read -a ARR_YML_CONFIGS <<< "$IN_YML_CONFIGS"

for CONFIG in "${ARR_YML_CONFIGS[@]}"
do
  FULL_PATH="$S2S_PREFIX/$CONFIG"
  YML_CONFIGS="$FULL_PATH,$YML_CONFIGS"
done

# Remove last comma
YML_CONFIGS="${YML_CONFIGS:0:${#YML_CONFIGS}-1}"
OUT_BASE_PATH='../results'
OUT_PATH="$OUT_BASE_PATH/$RESULTS_NAME"

# remove potentially old data
if [ -d $OUT_PATH ]; then
  if [ -z ${FORCE+x} ]; then
    echoerr "The directory $OUT_PATH exists and FORCE is not set, exiting!"
    exit 2
  else
    rm -rf $OUT_PATH/
    mkdir -p $OUT_PATH
  fi
fi

# Log everything to stderr
set -x

LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4" \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64 \
python -m bin.train --config_paths=$YML_CONFIGS \
                    --output_dir=$OUT_PATH 2>&1 | tee ../logs/tf.log
cd ..
