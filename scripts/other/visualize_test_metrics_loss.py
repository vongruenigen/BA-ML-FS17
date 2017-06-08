import os
import sys
import json
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: visualize_test_metrics_loss.json <test-metrics-loss-file>')
    sys.exit(2)

metrics_loss_file = argv[0]

with open(metrics_loss_file, 'r') as f:
    metrics_loss = json.load(f)

legend_txts = []

plot_x_ticks = list(map(int, metrics_loss.keys()))
plot_y_loss_values = list(metrics_loss.values())

plt.plot(plot_x_ticks, plot_y_loss_values)
legend_txts.append('Test Loss')

plt.legend(legend_txts)
plt.show()
