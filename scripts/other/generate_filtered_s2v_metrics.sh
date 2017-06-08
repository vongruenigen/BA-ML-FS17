#!/bin/bash

TOP_N=$1

if [ -z "$TOP_N" ]; then
  echo "ERROR: Missing top-n argument"
  exit 2
fi

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

S2V_MODELS="
wiki
twitter
"

python filter_s2v_metrics_for_top_n_sentences.py 2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000/test_s2v_cosine_similarity_sent2vec_twitter_bigrams.json 2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000/test_predictions.csv 2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000/test_outputs_sentences_distribution.csv 5 2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000/test_s2v_cosine_similartiy_twitter_bigrams_filtered_top_10.json

for dirname in $opensubtitles_models
do
  ls "$dirname" || (echo "ERROR: Directory $dirname not found" && exit 2)
done

for dirname in $reddit_models
do
  ls "$dirname" || (echo "ERROR: Directory $dirname not found" && exit 2)
done

for S2V_MODEL in $S2V_MODELS
do
  for MODEL in $OPENSUBTITLES_MODELS
  do
    TEST_PREDS="$MODEL/test_predictions.csv"
    SENT_DISTS="$MODEL/test_outputs_sentences_distribution.csv"
    S2V_METRICS="$MODEL/test_s2v_cosine_similarity_sent2vec_${S2V_MODEL}_bigrams.json"
    FILTERED_S2V_METRICS="$MODEL/test_s2v_cosine_similarity_${S2V_MODEL}_bigrams_filtered_top_$TOP_N.json"

    echo "Starting to generate filtered s2v metrics results..."
    python filter_s2v_metrics_for_top_n_sentences.py $S2V_METRICS $TEST_PREDS $SENT_DISTS $TOP_N $FILTERED_S2V_METRICS
    sleep 10
  done

  for MODEL in $REDDIT_MODELS
  do
    TEST_PREDS="$MODEL/test_predictions.csv"
    SENT_DISTS="$MODEL/test_outputs_sentences_distribution.csv"
    S2V_METRICS="$MODEL/test_s2v_cosine_similarity_sent2vec_${S2V_MODEL}_bigrams.json"
    FILTERED_S2V_METRICS="$MODEL/test_s2v_cosine_similarity_${S2V_MODEL}_bigrams_filtered_top_$TOP_N.json"

    echo "Starting to generate filtered s2v metrics results..."
    python filter_s2v_metrics_for_top_n_sentences.py $S2V_METRICS $TEST_PREDS $SENT_DISTS $TOP_N $FILTERED_S2V_METRICS
  done

  FINAL_REDDIT_OUT="final_metrics/s2v_cosine_metric_${S2V_MODEL}_reddit_filtered_$TOP_N.json"
  FINAL_OPENSUBTITLES_OUT="final_metrics/s2v_cosine_metric_${S2V_MODEL}_opensubtitles_filtered_$TOP_N.json"

  python combine_test_metric_results.py avg $FINAL_REDDIT_OUT 2017-05-*reddit*/*cosine_similarity*$S2V_MODEL*_top_${TOP_N}.json
  python combine_test_metric_results.py avg $FINAL_OPENSUBTITLES_OUT 2017-05-03_*not_reversed*/*cosine_similarity*$S2V_MODEL*_top_${TOP_N}.json
done
