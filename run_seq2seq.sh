SEQ2SEQ_DIR_NAME="seq2seq"

if [ ! -f $SEQ2SEQ_DIR_NAME ]; then
  echo "Cloning google/seq2seq project from github and store it in $SEQ2SEQ_DIR_NAME..."
  git clone https://github.com/google/seq2seq.git $SEQ2SEQ_DIR_NAME
fi

cd $SEQ2SEQ_DIR_NAME
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64 \
python -m bin.train --config_paths="../configs/seq2seq/nmt_middle_256.yml,../configs/seq2seq/train_seq2seq.yml,../configs/seq2seq/input_pipeline/opensubtitles.yml" \
                    --output_dir="../results/seq2seq-opus-2016/"
cd ..
