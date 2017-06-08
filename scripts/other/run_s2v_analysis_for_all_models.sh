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

S2V_MODELS="
sent2vec_wiki_bigrams
sent2vec_twitter_bigrams
"

for dirname in $opensubtitles_models
do
  ls "$dirname" || (echo "ERROR: Directory $dirname not found" && exit 2)
done

for dirname in $reddit_models
do
  ls "$dirname" || (echo "ERROR: Directory $dirname not found" && exit 2)
done

for S2V_MODEL_NAME in $S2V_MODELS
do
  for MODEL in $OPENSUBTITLES_MODELS
  do
    PREDS_FILE="$MODEL/test_predictions.csv"
    S2V_GENERATED="$MODEL/test_s2v_generated_$S2V_MODEL_NAME.h5"
    S2V_EXPECTED="$MODEL/test_s2v_expected_$S2V_MODEL_NAME.h5"
    S2V_RESULTS_COSINE="$MODEL/test_s2v_cosine_similarity_$S2V_MODEL_NAME.json"
    S2V_RESULTS_EUCLIDEAN="$MODEL/test_s2v_euclidean_similarity_$S2V_MODEL_NAME.json"

    echo "Starting to analyse the s2v embeddings for the opensubitles model $MODEL..."
    python scripts/analyse_s2v_sequence_embeddings.py $S2V_EXPECTED $S2V_GENERATED $S2V_RESULTS_COSINE cosine
    python scripts/analyse_s2v_sequence_embeddings.py $S2V_EXPECTED $S2V_GENERATED $S2V_RESULTS_EUCLIDEAN euclidean
  done

  for MODEL in $REDDIT_MODELS
  do
    PREDS_FILE="$MODEL/test_predictions.csv"
    S2V_GENERATED="$MODEL/test_s2v_generated_$S2V_MODEL_NAME.h5"
    S2V_EXPECTED="$MODEL/test_s2v_expected_$S2V_MODEL_NAME.h5"
    S2V_RESULTS_COSINE="$MODEL/test_s2v_cosine_similarity_$S2V_MODEL_NAME.json"
    S2V_RESULTS_EUCLIDEAN="$MODEL/test_s2v_euclidean_similarity_$S2V_MODEL_NAME.json"

    echo "Starting to analyse the s2v embeddings for the reddit model $MODEL..."
    python scripts/analyse_s2v_sequence_embeddings.py $S2V_EXPECTED $S2V_GENERATED $S2V_RESULTS_COSINE cosine
    python scripts/analyse_s2v_sequence_embeddings.py $S2V_EXPECTED $S2V_GENERATED $S2V_RESULTS_EUCLIDEAN euclidean
  done
done
