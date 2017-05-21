import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import operator

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: Missing mandatory argument(s)')
    print('       (visualize_ngram_bar.py <ngram-csv-path> [<top-n=0 (all)> <mode=bar|dist>])')
    sys.exit(2)

top_n = 0
ngrams_path = argv[0]
mode = 'bar'

if len(argv) > 1:
    top_n = int(argv[1])

if len(argv) > 2:
    mode = argv[2]

ngrams = []

for i, line in enumerate(open(ngrams_path, 'r')):
    if i == 0:
        continue # skip headings
    elif i == (top_n+1) and i != 0:
        break
    else:
        line_parts = line.split(';')
        ngrams.append((line_parts[0], int(line_parts[1])))

if mode == 'bar':
    ngrams = list(sorted(ngrams, key=operator.itemgetter(1)))

ngram_txts = list(map(operator.itemgetter(0), ngrams))
ngram_vals = list(map(operator.itemgetter(1), ngrams))
ngrams_pos = np.arange(len(ngrams)) + 0.5

if mode == 'bar':
    plt.yticks(ngrams_pos, ngram_txts)
    plt.xlabel('Frequency')
    plt.barh(ngrams_pos, ngram_vals, color='lightblue')
elif mode == 'dist':
    plt.ylabel('Frequency')
    plt.bar(ngrams_pos, ngram_vals, color='lightblue')

plt.show()
