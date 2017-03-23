#!/bin/bash
SEQ2SEQ_DIR_NAME="seq2seq"

if [ ! -d $SEQ2SEQ_DIR_NAME ]; then
  echo "Cloning google/seq2seq project from github and store it in $SEQ2SEQ_DIR_NAME..."
  git clone https://github.com/google/seq2seq.git $SEQ2SEQ_DIR_NAME
fi

OUT_PATH="../results/seq2seq-opus-2016"

export S2S_PREFIX='../configs/seq2seq'

# remove potentially old data
rm -rf $OUT_PATH/*
mkdir -p $OUT_PATH

# Run tensorboard in the background
python -m tensorflow.tensorboard --port=8008 --logdir=$OUT_PATH &2> tensorboard.log

# Log everything to stderr
set -x

cd $SEQ2SEQ_DIR_NAME
LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4" \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64 \
python -m bin.train --config_paths="$S2S_PREFIX/nmt_2048.yml,$S2S_PREFIX/train_seq2seq.yml,\
                                    $S2S_PREFIX/input_pipeline/opensubtitles.yml,$S2S_PREFIX/metrics.yml" \
                    --output_dir=$OUT_PATH 2> logs/tf.log
cd ..
