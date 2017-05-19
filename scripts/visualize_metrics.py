import os
import sys
import json
import matplotlib.pyplot as plt

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: visualize_metrics.json <metrics-file> <metrics-key> [<value_mode=perc|abs> <plot_mode=bar|plot> <y_scale=linear|log>]')
    sys.exit(2)

metrics_file_path = argv[0]
metrics_key = argv[1]
value_mode = 'perc'
plot_mode = 'bar'
y_scale = 'linear'
metrics_content = None

if len(argv) > 2:
    value_mode = argv[2]

if len(argv) > 3:
    plot_mode = argv[3]

if len(argv) > 4:
    y_scale = argv[4]

with open(metrics_file_path, 'r') as f:
    metrics_content = json.load(f)

metrics = metrics_content

for k in metrics_key.split('/'):
    if k in metrics:
        metrics = metrics[k]
    else:
        print('ERROR: Key %s missing in metrics file' % metrics_key)
        sys.exit(2)

if type(metrics) is list:
    metrics = {i+1: v for i, v in enumerate(metrics)}

x_values = list(map(float, metrics.keys()))
y_values = list(map(float, metrics.values()))
width = 1

if x_values[1] < 1.1:
    x_values = list(map(lambda x: x*100.0, x_values))
    width *= 9

plt.xticks(x_values)

if plot_mode == 'bar':
    plt.yscale(y_scale)
    plt.bar(x_values, y_values, color='blue', width=width)
elif plot_mode == 'plot':
    plt.plot(x_values, y_values, color='blue')

if value_mode == 'perc':
    gca = plt.gca()
    gca.set_yticklabels(['{:.0f}%'.format(x*100) for x in gca.get_yticks()])

plt.show()
