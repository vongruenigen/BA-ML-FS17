import os
import sys
import json

from operator import itemgetter

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: combine_test_metric_results.py <key> <out-json> <file1, file2, ...>')
    sys.exit(2)

metric_key = argv[0]
out_file = argv[1]
metric_files = argv[2:]
metric_dict = {}

def get_step_count(f):
    file_parts = f.split('_')
    step_count_idx = file_parts.index('step')+1

    if step_count_idx == -1:
        print('ERROR: File %s has no step count in its name!')
        sys.exit(2)

    return int(file_parts[step_count_idx])

for metrics_f_name in metric_files:
    metrics = None

    with open(metrics_f_name, 'r') as f:
        metrics = json.load(f)

    step_count = get_step_count(f)
    step_value = metrics[metric_key]

    metric_dict[step_count] = step_value

comb_metrics = list(metric_dict.items())
comb_metrics = sorted(comb_metrics, key=itemgetter(0))

with open(out_file, 'w+') as f:
    json.dump(metric_dict, f, indent=4, sort_keys=True)
