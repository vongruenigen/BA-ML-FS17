import os
import sys
import math
import pydot
import operator
import networkx as nx
import matplotlib.pyplot as plt

from colour import Color
from collections import defaultdict

from graphviz import Digraph

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument(s)')
    print('       (visualize_ngram_graph.py <ngram-csv-path> <top-n=all> '
          '<out-file> [<engine=networkx|pydot> <path-texts>])')
    sys.exit(2)

top_n = 0
engine = 'networkx'
ngrams_path = argv[0]
top_n = int(argv[1])
out_file = argv[2]

path_sequences_file = None
path_sequences = None
colored_edges = []
alt_edge_color = Color('red')
use_colored_edges = False

if len(argv) > 3:
    engine = argv[3]

if len(argv) > 4:
    path_sequences_file = argv[4]
    use_colored_edges = True

ngrams = []

# Start by extracting the ngrams from the text file
for i, line in enumerate(open(ngrams_path, 'r')):
    if i == 0:
        continue # skip headings
    elif i == (top_n+1) and top_n != 0:
        break
    else:
        line_parts = line.split(';')
        ngrams.append((tuple(line_parts[0].split()), int(line_parts[1])))

ngrams = list(sorted(ngrams, key=operator.itemgetter(1), reverse=True))
ngram_freqs = {n: f for n, f in ngrams}
ngram_txts = list(map(operator.itemgetter(0), ngrams))

graph_edges = set()

def add_edge(left_node, right_node, edges, color=None):
    edges.add((left_node, right_node, color))

if path_sequences_file is not None:
    print('Loading texts for the paths...')
    path_sequences = []

    with open(path_sequences_file, 'r') as paths_f:
        for line in paths_f:
            line_parts = line.strip('\n').lower().split()
            line_bigrams = list(zip(*[line_parts[i:] for i in range(2)]))
            path_sequences.append(line_bigrams)

    print('Finished loading the texts for the paths!')

# Build the edges of the graph
print('Building edges...')

for ngram_outter in ngram_txts:
    for ngram_inner in ngram_txts:
        if ngram_outter == ngram_inner:
            continue
        else:
            if ngram_outter[-1] == ngram_inner[0]:
                add_edge(ngram_outter, ngram_inner, graph_edges)
            elif ngram_inner[0] == ngram_outter[-1]:
                add_edge(ngram_inner, ngram_outter, graph_edges)

graph_nodes = set(x[0] for x in graph_edges)
graph_nodes |= set(x[1] for x in graph_edges)

print('Finished building edges!')

if use_colored_edges:
    print('Coloring the built edges...')

    for seq in path_sequences:
        for edge in graph_edges:
            in_node, out_node, _ = edge

            if in_node in seq and out_node in seq and \
               seq.index(edge[0])+1 == seq.index(edge[1]):
               colored_edges.append((in_node, out_node))

graph_node_stats = defaultdict(lambda: {'in': 0, 'out': 0, 'diff': 0})

print('Collecting some stats about the graph...')

for edge in graph_edges:
    left_node, right_node, _ = edge
    graph_node_stats[left_node]['out'] += 1
    graph_node_stats[right_node]['in'] += 1

for node in graph_node_stats:
    count = graph_node_stats[node]
    diff = count['in'] - count['out']
    graph_node_stats[node]['diff'] = diff

node_diff_values = [x['diff'] for x in graph_node_stats.values()]

max_in_count = max([x['in'] for x in graph_node_stats.values()])
max_out_count = max([x['out'] for x in graph_node_stats.values()])

max_diff_count = max(node_diff_values)
min_diff_count = min(node_diff_values)

diff_positions = list(range(min_diff_count, max_diff_count+1))

print('Finished collecting stats!')

start_color = Color('blue')
middle_color = Color('white')
end_color = Color('red')

middle_idx = diff_positions.index(0)
first_half_colors = list(start_color.range_to(middle_color, middle_idx+1))
second_half_colors = list(middle_color.range_to(end_color,
                          abs(len(diff_positions)-middle_idx-1)))

node_colors = first_half_colors + second_half_colors

ngram_freq_sum = sum(ngram_freqs.values())
ngram_size = {}

scale_factor = math.log(len(graph_nodes))/math.log(5)
widths = [1/scale_factor for _ in range(len(graph_nodes))]

if engine == 'networkx':
    print('Using networkx for the drawing')

    for ngram in ngram_txts:
        ngram_size[ngram] = ((10/scale_factor)*len(ngrams))*(ngram_freqs[ngram]/ngram_freq_sum)

    ngram_graph = nx.DiGraph()
    ngram_graph.add_nodes_from(graph_nodes)
    ngram_graph.add_edges_from(list(map(lambda x: (x[0], x[1]), graph_edges)))
    ngram_graph_pos=nx.fruchterman_reingold_layout(ngram_graph, k=0.05, iterations=100)
    node_sizes = list(reversed(range(len(ngrams))))

    nx.draw(ngram_graph, ngram_graph_pos, node_size=0.01, width=widths)

    for i, ngram in enumerate(ngram_txts):
        if ngram in ngram_graph_pos:
            x, y = ngram_graph_pos[ngram]
            freq = ngram_freqs[ngram]

            diff = graph_node_stats[ngram]['diff']
            diff_idx = diff_positions.index(diff)

            plt.text(x, y, s=' '.join(ngram), size=ngram_size[ngram],
                     bbox=dict(facecolor=node_colors[diff_idx].get_rgb()),
                     horizontalalignment='center')

    # plt.show()
    plt.savefig(out_file)

elif engine == 'pydot':
    print('Using pydot for drawing the graph')

    ngram_graph = pydot.Dot(graph_type='digraph')

    ngram_nodes = {}
    ngram_edges = {}

    for ngram in ngram_txts:
        diff = graph_node_stats[ngram]['diff']
        diff_idx = diff_positions.index(diff)

        ngram_nodes[ngram] = pydot.Node(' '.join(ngram), style='filled',
                                        fillcolor=node_colors[diff_idx].get_web(),
                                        fontsize=0.1*ngram_freq_sum/ngram_freqs[ngram])

        ngram_graph.add_node(ngram_nodes[ngram])

    for edge in graph_edges:
        in_node = ngram_nodes[edge[0]]
        out_node = ngram_nodes[edge[1]]
        edge_color = None

        if use_colored_edges and (in_node, out_node) in colored_edges:
            edge_color = alt_edge_color.get_web()

        ngram_edges[edge] = pydot.Edge(in_node, out_node, penwidth=3)

        ngram_graph.add_edge(ngram_edges[edge])

    print('Starting to draw graph...')

    ngram_graph.set_outputorder('edgesfirst')

    if out_file.lower().endswith('.svg'):
        ngram_graph.write_svg(out_file, prog='fdp')
    elif out_file.lower().endswith('.png'):
        ngram_graph.write_png(out_file, prog='fdp')
    elif out_file.lower().endswith('.pdf'):
        ngram_graph.write_pdf(out_file, prog='fdp')
    elif out_file.lower().endswith('.ps'):
        ngram_graph.write_ps(out_file, prog='fdp')
    elif out_file.lower().endswith('.ps2'):
        ngram_graph.write_ps2(out_file, prog='fdp')
    elif out_file.lower().endswith('.dot'):
        ngram_graph.write_dot(out_file, prog='fdp')
    else:
        print('ERROR: Out file must be an SVG or PNG!')
        sys.exit(2)

    print('Stored graph in %s' % out_file)
