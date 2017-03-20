#!/bin/bash
SEQ2SEQ_DIR_NAME="seq2seq"

if [ ! -f $SEQ2SEQ_DIR_NAME ]; then
  echo "Cloning google/seq2seq project from github and store it in $SEQ2SEQ_DIR_NAME..."
  git clone https://github.com/google/seq2seq.git $SEQ2SEQ_DIR_NAME
fi

if [ -f /.dockerenv ]; then
  pip install cython
  pip install -r /BA-ML-FS17/requirements.txt
fi

export S2S_PREFIX='../configs/seq2seq'

cd $SEQ2SEQ_DIR_NAME
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64 \
python -m bin.train --config_paths="$S2S_PREFIX/nmt_4096.yml,$S2S_PREFIX/train_seq2seq.yml,\
                                    $S2S_PREFIX/input_pipeline/opensubtitles.yml,$S2S_PREFIX/metrics.yml" \
                    --output_dir="../results/seq2seq-opus-2016/"
cd ..
