import os
import sys
import json
import math
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: visualize_train_metrics.json <train-metrics-file>')
    sys.exit(2)

metrics_file_path = argv[0]

with open(metrics_file_path, 'r') as f:
    metrics_content = json.load(f)

metrics = metrics_content
legend_txts = []

train_metrics_loss = metrics['loss']
train_metrics_pplx = metrics['perplexity']
train_x_ticks = list(range(1, len(train_metrics_loss)+1))

val_metrics_loss = metrics['val_loss']
val_metrics_pplx = metrics['val_perplexity']
val_x_step_size = math.floor(len(train_metrics_loss)/len(val_metrics_loss))
val_x_ticks = list(range(0, len(train_metrics_loss), val_x_step_size))

plot_x_ticks = [0]

for i, tick in enumerate(train_x_ticks):
    if (i+1) % 1000 == 0:
        plot_x_ticks.append(tick)

plt.ylim(0,50)
plt.xticks(plot_x_ticks, rotation='vertical')

train_x_ticks = train_x_ticks[:len(train_metrics_loss)]
val_x_ticks = val_x_ticks[:len(val_metrics_loss)]

plt.plot(train_x_ticks, train_metrics_loss)
legend_txts.append('Train Loss')

plt.plot(train_x_ticks, train_metrics_pplx)
legend_txts.append('Train Perplexity')

plt.plot(val_x_ticks, val_metrics_loss)
legend_txts.append('Validation Loss')

plt.plot(val_x_ticks, val_metrics_pplx)
legend_txts.append('Validation Perplexity')

plt.legend(legend_txts)
plt.show()#('plot.png')
