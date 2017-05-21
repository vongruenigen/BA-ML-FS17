import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict

argv = sys.argv[1:]

if len(argv) < 1:
    print('ERROR: Missing mandatory argument(s)')
    print('       (visualize_ngram_graph.py <ngram-path> [<top-n=all>])')
    sys.exit(2)

top_n = 0
ngrams_path = argv[0]

if len(argv) > 1:
    top_n = int(argv[1])

ngrams = [tuple(l.split()) for l in open(ngrams_path, 'r')]

if top_n == 0 or top_n > len(ngrams):
    top_n = len(ngrams)

ngrams = ngrams[:top_n]

graph_edges = set()
add_edge = lambda x, y: graph_edges.add((x, y))

for ngram_outter in ngrams:
    if len(graph_edges) == top_n:
        break

    for ngram_inner in ngrams:
        if len(graph_edges) == top_n:
            break
        if ngram_outter == ngram_inner:
            continue
        else:
            if ngram_outter[1] == ngram_inner[0]:
                add_edge(ngram_outter, ngram_inner)
            elif ngram_inner[1] == ngram_outter[0]:
                add_edge(ngram_inner, ngram_outter)

graph_nodes = set(x[0] for x in graph_edges)
graph_nodes |= set(x[1] for x in graph_edges)

ngram_graph = nx.Graph()
ngram_graph.add_nodes_from(graph_nodes)
ngram_graph.add_edges_from(graph_edges)
ngram_graph_pos=nx.spring_layout(ngram_graph)
node_sizes = list(reversed(range(len(ngrams))))

nx.draw(ngram_graph, ngram_graph_pos, cmap=plt.cm.Blues, 
        node_size=node_sizes)

for ngram in ngrams:
    if ngram in ngram_graph_pos:
        x, y = ngram_graph_pos[ngram]
        plt.text(x, y, s=' '.join(ngram), bbox=dict(facecolor='lightblue'),
                 horizontalalignment='center')

plt.show()

# ngram.

