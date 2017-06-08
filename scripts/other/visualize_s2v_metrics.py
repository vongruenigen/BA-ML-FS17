import os
import sys
import json
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: visualize_s2v_metrics.json <s2v-metrics-file>')
    sys.exit(2)

s2v_metrics_file = argv[0]

with open(s2v_metrics_file, 'r') as f:
    metrics = json.load(f)

legend_txts = []

plot_x_ticks = list(map(int, metrics.keys()))
plot_y_values = list(metrics.values())

plt.plot(plot_x_ticks, plot_y_values)
legend_txts.append('Avg. Similarity Sent2Vec')

plt.ylim(0, 0.5)

plt.legend(legend_txts)
plt.show()
