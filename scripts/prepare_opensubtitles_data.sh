#!/bin/bash
python scripts/preprocess_opensubtitles_data.py data/opensubtitles/raw_2016 \
                                                data/opensubtitles/opensubtitles_raw_all_2016.txt

python scripts/split_corpus.py data/opensubtitles/opensubtitles_raw_all_2016.txt \
                               70,20,10 data/opensubtitles/opensubtitles_2016_train.txt \
                               data/opensubtitles/opensubtitles_2016_valid.txt \
                               data/opensubtitles/opensubtitles_2016_test.txt 10

export END=9

for i in $(seq 0 $END); do
  python scripts/convert_corpus_to_parallel_text_format.py \
          data/opensubtitles/opensubtitles_2016_train.$i.txt \
          data/opensubtitles/opensubtitles_2016_train.$i.src.txt \
          data/opensubtitles/opensubtitles_2016_train.$i.target.txt

  python scripts/convert_corpus_to_parallel_text_format.py \
          data/opensubtitles/opensubtitles_2016_valid.$i.txt \
          data/opensubtitles/opensubtitles_2016_valid.$i.src.txt \
          data/opensubtitles/opensubtitles_2016_valid.$i.target.txt

  python scripts/convert_corpus_to_parallel_text_format.py \
          data/opensubtitles/opensubtitles_2016_test.$i.txt \
          data/opensubtitles/opensubtitles_2016_test.$i.src.txt \
          data/opensubtitles/opensubtitles_2016_test.$i.target.txt
done
