import os
import sys
import json
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: visualize_test_metrics.json <test-metrics-loss-file> <test-metrics-perplexity-file>')
    sys.exit(2)

metrics_loss_file = argv[0]
metrics_pplx_file = argv[1]

with open(metrics_loss_file, 'r') as f:
    metrics_loss = json.load(f)

with open(metrics_pplx_file, 'r') as f:
    metrics_pplx = json.load(f)

legend_txts = []

plot_x_ticks = list(map(int, metrics_loss.keys()))
plot_y_loss_values = list(metrics_loss.values())
plot_y_pplx_values = list(metrics_pplx.values())

plt.plot(plot_x_ticks, plot_y_loss_values)
legend_txts.append('Test Loss')

plt.plot(plot_x_ticks, plot_y_pplx_values)
legend_txts.append('Test Perplexity')

plt.legend(legend_txts)
plt.show()
