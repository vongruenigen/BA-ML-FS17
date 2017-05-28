import os
import sys
import h5py
import json
import scipy
import math
import numpy as np

from scipy.spatial import distance

from tqdm import tqdm

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument')
    print('       (analyse_s2v_sequence_embddings.py <output-h5> '
          '<expected-h5> <results-json> [<metric=cosine|euclidean>])')
    sys.exit(2)

output_h5, expected_h5, results_json = argv[:3]
metric = 'cosine'

if len(argv) > 3:
    metric = argv[3]

euclidean_fn = lambda x, y: distance.euclidean(x, y)
cosine_fn = lambda x, y: 1 - distance.cosine(x, y)

metric_map = {'cosine': cosine_fn, 'euclidean': euclidean_fn}

if metric not in metric_map:
    print('ERROR: Metrics "%s" does not exist!' % metric)
    sys.exit(2)

with h5py.File(output_h5) as out_f:
    with h5py.File(expected_h5) as exp_f:
        with open(results_json, 'w') as results_f:
            exp_ds = exp_f['embeddings']
            out_ds = out_f['embeddings']
            assert(len(exp_ds) == len(out_ds))

            sample_results = []
            metric_fn = metric_map[metric]
            iterator = tqdm(enumerate(zip(out_ds, exp_ds)), total=len(exp_ds))

            for i, (out_sample, exp_sample) in iterator:
                sample_results.append(metric_fn(out_sample, exp_sample))

            sample_results = list(map(lambda x: 0 if math.isnan(x) else x, sample_results))
            avg_result = float(sum(sample_results)) / len(sample_results)
            results_dict = {'metric': metric, 'avg': avg_result, 'per_sample': sample_results}
            json.dump(results_dict, results_f, indent=4, sort_keys=True)

            print('Finished analyzing the embeddings for their similarities!')
