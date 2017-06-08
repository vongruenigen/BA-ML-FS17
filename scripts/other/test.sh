#!/bin/bash
opensubtitles_models="
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

reddit_models="
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_505000/model-0.chkp-505000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_1000000/model-0.chkp-1000000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_1500000/model-0.chkp-1500000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_2005000/model-0.chkp-2005000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_2500000/model-0.chkp-2500000
/media/dvg/Volume/ZHAW/BA/trained-models/2017-05-03_15-27-16_new_reddit_movies_videos_tele_2008-2015_step_3070000/model-0.chkp-3070000
"

for filename in $opensubtitles_models
do
  ls "$filename.data-00000-of-00001" || (echo "ERROR: File $filename not found" && exit 2)
done

for filename in $reddit_models
do
  ls "$filename.data-00000-of-00001" || (echo "ERROR: File $filename not found" && exit 2)
done
