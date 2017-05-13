#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#

import sys
import os
import helpers
import numpy as np
import matplotlib.pyplot as plt

helpers.expand_import_path_to_source()

from os import path
from runner import Runner
from config import Config
from sklearn.decomposition import PCA

argv = sys.argv[1:]

if len(argv) < 2:
    print('ERROR: Missing mandatory argument!')
    print('       (python scripts/visualize_attention.py <model-path> <input-txt>)')
    sys.exit(2)

model_path = argv[0]
input_txt = argv[1]

INPUT_EMB_NAME = 'model_with_buckets/embedding_attention_seq2seq/rnn/concat_49:0'

if not path.isfile(input_txt):
    print('ERROR: The input-txt param has to point to a file')
    sys.exit(2)

model_dir = '/'.join(model_path.split('/')[:-1])
input_seqs = [line for line in open(input_txt, 'r')]
config_path = path.join(model_dir, 'config.json')
chkp_path = path.join(model_dir, 'checkpoint')

config = Config.load_from_json(config_path).for_inference()
config.set('model_path', model_path)

# TODO: Currently only works in greedy mode!
config.set('use_beam_search', False)

runner = Runner(config)
tokenize = runner.data_loader.get_tokenizer()
results = []

for i, input_seq in enumerate(input_seqs):
    prediction, input_emb = runner.inference(input_seq, additional_tensor_names=[INPUT_EMB_NAME])
    input_emb = input_emb[0].reshape(-1)[-config.get('num_hidden_units'):]
    results.append((input_seq, prediction, input_emb))
    print('Finished prediction #%d...' % (i+1))

reverse_input = config.get('reverse_input')
input_embeddings = [results[i][-1] for i in range(len(results))]
input_seqs = [r[0] for r in results]

pca = PCA(n_components=2)
np.set_printoptions(suppress=True)
proj_inp = pca.fit_transform(input_embeddings)

x_min = min(proj_inp[:, 0])
x_max = max(proj_inp[:, 0])
y_min = min(proj_inp[:, 1])
y_max = max(proj_inp[:, 1])

plt.axis([x_min-(x_max/2), x_max+(x_max/2), y_min-(y_max/2), y_max+(y_max/2)])
plt.scatter(proj_inp[:, 0], proj_inp[:, 1])

for inp_seq, x, y in zip(input_seqs, proj_inp[:, 0], proj_inp[:, 1]):
    plt.annotate(inp_seq, xy=(x, y), xytext=(0, 0),
                 textcoords='offset points', fontsize=5)

plt.show()
