#!/bin/bash

TOP_N=250

OPENSUBTITLES_MODELS="
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_495000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1525000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_3005000
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
  CORPUS_BIGRAM='ngram_analysis/opensubitles_2016_bigram_collocation_likelihood.csv'
  CORPUS_UNIGRAM='ngram_analysis/reddit_unigram_collocation_likelihood.csv'
  GENERATED_BIGRAM="$MODEL/test_outputs_bigrams.csv"
  GENERATED_UNIGRAM="$MODEL/test_outputs_bigrams_words.csv"

  echo "Starting to generate comparison plots for the ngram distributions..."

  python compare_ngram_distributions.py $CORPUS_BIGRAM $GENERATED_BIGRAM $TOP_N
  mv ngram_distribution_comparison.pdf "$MODEL/bigram_distribution_comparison.pdf"

  python compare_ngram_distributions.py $CORPUS_UNIGRAM $GENERATED_UNIGRAM $TOP_N
  mv ngram_distribution_comparison.pdf "$MODEL/unigram_distribution_comparison.pdf"
done

for MODEL in $REDDIT_MODELS
do
  CORPUS_BIGRAM='ngram_analysis/reddit_bigram_collocation_likelihood.csv'
  CORPUS_UNIGRAM='ngram_analysis/reddit_unigram_collocation_likelihood.csv'
  GENERATED_FILE="$MODEL/test_outputs_bigrams.csv"
  GENERATED_UNIGRAM="$MODEL/test_outputs_bigrams_words.csv"

  echo "Starting to generate comparison plots for the ngram distributions..."
  
  python compare_ngram_distributions.py $CORPUS_BIGRAM $GENERATED_FILE $TOP_N
  mv ngram_distribution_comparison.pdf "$MODEL/bigram_distribution_comparison.pdf"

  python compare_ngram_distributions.py $CORPUS_UNIGRAM $GENERATED_UNIGRAM $TOP_N
  mv ngram_distribution_comparison.pdf "$MODEL/unigram_distribution_comparison.pdf"
done
