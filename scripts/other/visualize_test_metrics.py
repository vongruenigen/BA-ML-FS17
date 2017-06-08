import os
import sys
import json
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: visualize_test_metrics.json <opus-test-metrics> <reddit-test-metrics> <metric-name>')
    sys.exit(2)

opus_test_metrics = argv[0]
reddit_test_metrics = argv[1]
metric_name = argv[2]

opus_metric_values = None
reddit_metric_values = None

with open(opus_test_metrics, 'r') as f:
    opus_metric_values = json.load(f)

with open(reddit_test_metrics, 'r') as f:
    reddit_metric_values = json.load(f)

metrics_all = [opus_metric_values, reddit_metric_values]
legend_txts = ('OpenSubtitles', 'Reddit')

for metrics, name in zip(metrics_all, legend_txts):
    plot_x_ticks = list(map(int, metrics.keys()))
    plot_y_values = list(metrics.values())
    plt.plot(plot_x_ticks, plot_y_values)

plt.legend(legend_txts)
plt.show()
