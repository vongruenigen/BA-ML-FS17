#!/bin/bash

# Loss & perplexity metrics from testing
python combine_test_metric_results.py loss final_metrics/test_metric_results_opensubtitles_not_reversed_loss.json 2017-05-03_*not_reversed*/test_metrics.json
python combine_test_metric_results.py perplexity final_metrics/test_metric_results_opensubtitles_not_reversed_perplexity.json 2017-05-03_*not_reversed*/test_metrics.json
python combine_test_metric_results.py loss final_metrics/test_metric_results_opensubtitles_loss.json 2017-05-03_*2048_step*/test_metrics.json
python combine_test_metric_results.py perplexity final_metrics/test_metric_results_opensubtitles_perplexity.json 2017-05-03_*2048_step*/test_metrics.json
python combine_test_metric_results.py loss final_metrics/test_metric_results_reddit_loss.json 2017-05-03_*reddit*/test_metrics.json
python combine_test_metric_results.py perplexity final_metrics/test_metric_results_reddit_perplexity.json 2017-05-03_*reddit*/test_metrics.json

# Cosine & euclidean metrics from s2v analysis (twitter)
python combine_test_metric_results.py avg final_metrics/s2v_cosine_metric_results_opensubtitles_not_reversed.json 2017-05-03_*not_reversed*/test_s2v_cosine*sent2vec_twitter*.json
python combine_test_metric_results.py avg final_metrics/s2v_euclidean_metric_results_opensubtitles_not_reversed.json 2017-05-03_*not_reversed*/test_s2v_euclidean*sent2vec_twitter*.json
python combine_test_metric_results.py avg final_metrics/s2v_cosine_metric_opensubtitles.json 2017-05-03_*2048_step*/test_s2v_cosine*sent2vec_twitter*.json
python combine_test_metric_results.py avg final_metrics/s2v_euclidean_results_opensubtitles.json 2017-05-03_*2048_step*/test_s2v_euclidean*sent2vec_twitter*.json
python combine_test_metric_results.py avg final_metrics/s2v_cosine_metric_reddit.json 2017-05-03_*reddit*/test_s2v_cosine*sent2vec_twitter*.json
python combine_test_metric_results.py avg final_metrics/s2v_euclidean_results_reddit.json 2017-05-03_*reddit*/test_s2v_euclidean*sent2vec_twitter*.json

# Cosine & euclidean metrics from s2v analysis (wiki)
python combine_test_metric_results.py avg final_metrics/s2v_cosine_metric_results_opensubtitles_not_reversed.json 2017-05-03_*not_reversed*/test_s2v_cosine*sent2vec_wiki*.json
python combine_test_metric_results.py avg final_metrics/s2v_euclidean_metric_results_opensubtitles_not_reversed.json 2017-05-03_*not_reversed*/test_s2v_euclidean*sent2vec_wiki*.json
python combine_test_metric_results.py avg final_metrics/s2v_cosine_metric_opensubtitles.json 2017-05-03_*2048_step*/test_s2v_cosine*sent2vec_wiki*.json
python combine_test_metric_results.py avg final_metrics/s2v_euclidean_results_opensubtitles.json 2017-05-03_*2048_step*/test_s2v_euclidean*sent2vec_wiki*.json
python combine_test_metric_results.py avg final_metrics/s2v_cosine_metric_reddit.json 2017-05-03_*reddit*/test_s2v_cosine*sent2vec_wiki*.json
python combine_test_metric_results.py avg final_metrics/s2v_euclidean_results_reddit.json 2017-05-03_*reddit*/test_s2v_euclidean*sent2vec_wiki*.json

# Training metrics
cp 2017-05-03_*not_reversed*3005000/metrics.json final_metrics/traing_metrics_opensubtitles_not_reversed.json
cp 2017-05-03_*2048_step*3005000/metrics.json final_metrics/traing_metrics_opensubtitles.json
cp 2017-05-03_*reddit*3070000/metrics.json final_metrics/training_metrics_reddit.json
