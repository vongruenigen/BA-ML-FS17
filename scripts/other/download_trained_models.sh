#!/bin/bash

TARGET_PATH="/media/dvg/Volume/ZHAW/BA/trained-models/"
DIRNAME="~/trained-models"

REDDIT_PREFIX="reddit-new"
REDDIT_MODELS=""

OPENSUBTITLES_PREFIX="opensubtitles-2016-new"
OPENSUBTITLES_MODELS="
2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_2505000.zip
2017-05-03_08-59-10_opensubtitles_2016_large_2048_not_reversed_step_3005000.zip
"

echo "Downloading reddit models"

for MODEL in $REDDIT_MODELS
do
  MODEL_PATH="$DIRNAME/$REDDIT_PREFIX/$MODEL"
  echo "Syncing the model $MODEL"
  
  while true
  do
    echo "Starting rsync..."
    timeout 30 rsync --partial --progress --rsh=ssh weilemar@srv-lab-t-697:$MODEL_PATH .

    if [ "$?" = "0" ]
    then
      mv $MODEL $TARGET_PATH
      echo "Finished syncing the model $MODEL"
      break
    fi
  done
done

echo "Downloading OpenSubtitles models"

for MODEL in $OPENSUBTITLES_MODELS
do
  MODEL_PATH="$DIRNAME/$OPENSUBTITLES_PREFIX/$MODEL"
  echo "Syncing the model $MODEL"
  
  while true
  do
    echo "Starting rsync..."
    timeout 15 rsync --partial --progress --rsh=ssh weilemar@srv-lab-t-697:$MODEL_PATH .

    if [ "$?" = "0" ]
    then
      mv $MODEL $TARGET_PATH
      echo "Finished syncing the model $MODEL"
      break
    fi
  done
done

