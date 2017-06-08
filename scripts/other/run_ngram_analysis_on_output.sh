#!/bin/bash

TOP_N_SAMPLES=250000

OPENSUBTITLES_MODELS="
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_495000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1525000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_3005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_1035000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_1505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_2505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_3005000
"

REDDIT_MODELS="
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_1000000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_1500000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_2500000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_3070000
"

for dirname in $opensubtitles_models
do
  ls "$dirname" || (echo "ERROR: Directory $dirname not found" && exit 2)
done

for dirname in $reddit_models
do
  ls "$dirname" || (echo "ERROR: Directory $dirname not found" && exit 2)
done

for MODEL in $OPENSUBTITLES_MODELS
do
  GENERATED_FILE="$MODEL/test_outputs.txt"
  GENERATED_BIGRAM_FILE="$MODEL/test_outputs_bigrams.csv"
  GENERATED_BIGRAM_WORDS_FILE="$MODEL/test_outputs_bigrams_words.csv"
  EXPECTED_FILE="$MODEL/test_expected.txt"

  echo "Starting to analyse the bigrams for the outputs of the model $MODEL..."
  python scripts/analyse_ngrams_from_corpus.py $GENERATED_FILE 2 $GENERATED_BIGRAM_FILE $GENERATED_BIGRAM_WORDS_FILE
done

for MODEL in $REDDIT_MODELS
do
  GENERATED_FILE="$MODEL/test_outputs.txt"
  GENERATED_BIGRAM_FILE="$MODEL/test_outputs_bigrams.csv"
  GENERATED_BIGRAM_WORDS_FILE="$MODEL/test_outputs_bigrams_words.csv"
  EXPECTED_FILE="$MODEL/test_expected.txt"

  echo "Starting to analyse the bigrams for the outputs of the model $MODEL..."
  python scripts/analyse_ngrams_from_corpus.py $GENERATED_FILE 2 $GENERATED_BIGRAM_FILE $GENERATED_BIGRAM_WORDS_FILE
done
