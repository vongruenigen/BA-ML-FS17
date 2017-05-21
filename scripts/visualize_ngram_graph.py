import os
import sys
import operator
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: Missing mandatory argument(s)')
    print('       (visualize_ngram_graph.py <ngram-csv-path> [<top-n=all>])')
    sys.exit(2)

top_n = 0
ngrams_path = argv[0]

if len(argv) > 1:
    top_n = int(argv[1])

ngrams = []

for i, line in enumerate(open(ngrams_path, 'r')):
    if i == 0: continue # skip headings
    else:
        line_parts = line.split(';')
        ngrams.append((line_parts[0], int(line_parts[1])))

ngrams = list(sorted(ngrams, key=operator.itemgetter(1), reverse=True))

if top_n == 0 or top_n > len(ngrams):
    top_n = len(ngrams)

ngrams = ngrams[:top_n]
ngram_freqs = {tuple(n.split()): f for n, f in ngrams}
ngram_txts = list(map(lambda x: tuple(x[0].split()), ngrams))

graph_edges = set()
add_edge = lambda x, y: graph_edges.add((x, y))

for ngram_outter in ngram_txts:
    for ngram_inner in ngram_txts:
        if ngram_outter == ngram_inner:
            continue
        else:
            if ngram_outter[1] == ngram_inner[0]:
                add_edge(ngram_outter, ngram_inner)
            elif ngram_inner[0] == ngram_outter[1]:
                add_edge(ngram_inner, ngram_outter)

graph_nodes = set(x[0] for x in graph_edges)
graph_nodes |= set(x[1] for x in graph_edges)

ngram_graph = nx.Graph()
ngram_graph.add_nodes_from(graph_nodes)
ngram_graph.add_edges_from(graph_edges)
ngram_graph_pos=nx.spring_layout(ngram_graph)
node_sizes = list(reversed(range(len(ngrams))))

nx.draw(ngram_graph, ngram_graph_pos, node_size=0.01, cmap=plt.cm.Blues)

ngram_freq_sum = sum(ngram_freqs.values())
ngram_size = {}

for ngram in ngram_txts:
    ngram_size[ngram] = (10*len(ngrams))*(ngram_freqs[ngram]/ngram_freq_sum)

for ngram in ngram_txts:
    if ngram in ngram_graph_pos:
        x, y = ngram_graph_pos[ngram]
        freq = ngram_freqs[ngram]
        plt.text(x, y, s=' '.join(ngram), size=ngram_size[ngram],
                 bbox=dict(facecolor='lightblue', zorder=freq),
                 horizontalalignment='center')

plt.show()

# ngram.

