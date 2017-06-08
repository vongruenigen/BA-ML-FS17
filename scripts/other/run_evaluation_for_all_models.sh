#!/bin/bash

TOP_N_SAMPLES=250000

OPENSUBTITLES_MODELS="
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_495000/model-0.chkp-495000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1035000/model-0.chkp-1035000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_1525000/model-0.chkp-1525000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2005000/model-0.chkp-2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2505000/model-0.chkp-2505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_3005000/model-0.chkp-3005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_505000/model-0.chkp-505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_1035000/model-0.chkp-1035000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_1505000/model-0.chkp-1505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_2005000/model-0.chkp-2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_2505000/model-0.chkp-2505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_09-03-23_opensubtitles_2016_large_2048_step_3005000/model-0.chkp-3005000
"

REDDIT_MODELS="
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_505000/model-0.chkp-505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_1000000/model-0.chkp-1000000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_1500000/model-0.chkp-1500000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_2005000/model-0.chkp-2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_2500000/model-0.chkp-2500000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_3070000/model-0.chkp-3070000
"

for filename in $OPENSUBTITLES_MODELS
do
  ls "$filename.data-00000-of-00001" || (echo "ERROR: File $filename not found" && exit 2)
done

for filename in $REDDIT_MODELS
do
  ls "$filename.data-00000-of-00001" || (echo "ERROR: File $filename not found" && exit 2)
done

for MODEL in $OPENSUBTITLES_MODELS
do
  MODEL_DIRNAME=`dirname "$MODEL.data-00000-of-00001"`
  METRICS_FILE="$MODEL_DIRNAME/test_metrics.json"
  PREDS_FILE="$MODEL_DIRNAME/test_predictions.csv"

  if [ -f "$PREDS_FILE" ]; then
    echo "Skipping $MODEL as the prediction file already exists!"
    continue
  fi
  
  echo "Starting to evaluate the opensubitles model $MODEL, storing results in $METRICS_FILE..."
  python scripts/evaluate_trained_model.py $MODEL \
                                           data/opensubtitles-2016/opensubtitles_2016.test.0.txt \
                                           $METRICS_FILE \
                                           $PREDS_FILE \
                                           $TOP_N_SAMPLES
done

for MODEL in $REDDIT_MODELS
do
  MODEL_DIRNAME=`dirname "$MODEL.data-00000-of-00001"`
  METRICS_FILE="$MODEL_DIRNAME/test_metrics.json"
  PREDS_FILE="$MODEL_DIRNAME/test_predictions.csv"

  if [ -f "$PREDS_FILE" ]; then
    echo "Skipping $MODEL as the prediction file already exists!"
    continue
  fi

  echo "Starting to evaluate the reddit model $MODEL..."
  python scripts/evaluate_trained_model.py $MODEL \
                                           data/new_reddit_dialog_movies_videos_television/reddit_tree_vid-tel-mov_2008-2015_test.0.txt \
                                           $METRICS_FILE \
                                           $PREDS_FILE \
                                           $TOP_N_SAMPLES
done

echo "Finished to evaluate all models!"
