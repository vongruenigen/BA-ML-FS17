#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#

import sys
import os
import helpers

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
out_type = 'overlay'

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
results = []

for i, input_seq in enumerate(input_seqs):
    prediction, attn_weights = runner.inference(input_seq,
                                                additional_tensor_names=attn_weights_names)
    results.append({'input': input_seq,
                    'output': prediction,
                    'attn_weights': attn_weights})

    print('Finished prediction #%d...' % (i+1))

import pdb; pdb.set_trace()

if out_type == 'overlay':
    pass
elif out_type == 'heatmap':
    pass
