import os
import sys
import json
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 5:
    print('ERROR: filter_s2v_metrics_for_top_n_sentences.json <s2v-metrics-file> <test-outputs-csv> <sentences-count-csv> <exclude-top-n> <out-json>')
    sys.exit(2)

s2v_metrics_file = argv[0]
test_outputs_file = argv[1]
sentences_count_file = argv[2]
exclude_top_n = int(argv[3])
out_json_path = argv[4]

with open(s2v_metrics_file, 'r') as f:
    metrics = json.load(f)

sentences_to_exclude = []

with open(sentences_count_file, 'r') as sent_f:
    for i, line in enumerate(sent_f):
        if i == 0:
            continue # skip headings
        else:
            line_parts = line.split(';')
            sentences_to_exclude.append(line_parts[0])

            if len(sentences_to_exclude) == exclude_top_n:
                break

exclude_output_idxs = set()

with open(test_outputs_file, 'r') as test_f:
    for i, line in enumerate(test_f):
        if i == 0:
            continue # skip headings
        else:
            line_parts = line.strip('\n').split(';')

            if line_parts[2] in sentences_to_exclude:
                exclude_output_idxs.add(i)

metric_values = metrics['per_sample']
new_metric_values = []

for i, v in enumerate(metric_values):
    if i not in exclude_output_idxs:
        new_metric_values.append(v)

metric_avg = sum(new_metric_values) / len(new_metric_values)

with open(out_json_path, 'w+') as out_f:
    out_dict = {'avg': metric_avg, 'per_sample': new_metric_values, 'metric': metrics['metric']}
    json.dump(out_dict, out_f, indent=4, sort_keys=True)

print('%f from the previous sample remain in the new metrics file' % (len(new_metric_values) / len(metrics['per_sample'])))
