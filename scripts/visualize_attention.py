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

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument!')
    print('       (python scripts/visualize_attention.py <modelpath> <input-txt> <output-dir> [<out-type=overlay>])')
    sys.exit(2)

model_path = argv[0]
input_txt = argv[1]
output_dir = argv[2]
possible_out_types =  ['overlay', 'heatmap']
out_type = 'heatmap'

if len(argv) > 3:
    out_type = argv[4]

if out_type not in possible_out_types:
    print('ERROR: out-type has to be one of %s', ', '.join(possible_out_types))
    sys.exit(2)

ATTN_W_VAR_NAME = 'model_with_buckets/embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/Attention_0_%d/Softmax:0'
attn_weights_names = [ATTN_W_VAR_NAME % i for i in range(1, 30)]

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
    prediction, attn_weights = runner.inference(input_seq,
                                                additional_tensor_names=attn_weights_names)
    results.append((tokenize(input_seq),
                    tokenize(prediction),
                    attn_weights))

    print('Finished prediction #%d...' % (i+1))

reverse_input = config.get('reverse_input')

if out_type == 'overlay':
    print('ERROR: Not implemented yet')
    sys.exit(2)
elif out_type == 'heatmap':
    out_file_name = path.join(output_dir, 'attention_visualization_%d.png')

    for i, (input_seq, output_seq, attn_weights) in enumerate(results):
        heatmap_val = np.zeros((len(output_seq), len(input_seq)))

        for j in range(1, heatmap_val.shape[0]):
            heatmap_val[j,:] = attn_weights[j][0,:len(input_seq)]

        if reverse_input:
            heatmap_val = np.flip(heatmap_val, 1)

        fig, ax = plt.subplots()
        ax.yaxis.labelpad = 20
        ax.xaxis.tick_top()
        ax.invert_yaxis()

        heatmap_obj = ax.imshow(heatmap_val, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(heatmap_obj)

        x_ticks = np.arange(0, len(input_seq))
        x_ticks_txt = input_seq

        y_ticks = np.arange(0, len(output_seq))
        y_ticks_txt = output_seq

        ax.set_xticklabels('')
        ax.set_yticklabels('')

        ax.set_xticks(x_ticks)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks_txt, ha='right')

        if 'SHOW' in os.environ:
            plt.show()
        else:
            plt.savefig(out_file_name % i)
