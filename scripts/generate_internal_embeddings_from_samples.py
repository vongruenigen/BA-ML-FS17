import os
import sys
import helpers
import h5py

from tqdm import tqdm
from os import path

helpers.expand_import_path_to_source()

from runner import Runner
from config import Config

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: generate_internal_embeddings_from_samples.py <samples-txt> <model-path> <embeddings-out>')
    sys.exit(2)

EMB_TENSOR_NAME = 'model_with_buckets/embedding_attention_seq2seq/rnn/basic_lstm_cell_29/concat:0'

sample_txts_path = argv[0]
model_path = argv[1]
embs_out_path = argv[2]

if path.isfile(embs_out_path):
    msg = 'Output file at "%s" already exists, delete? (y/n): ' % embs_out_path

    if input(msg).strip('\n').lower() == 'y':
        os.remove(embs_out_path)
    else:
        print('Quitting')
        sys.exit(2)


model_dir = '/'.join(model_path.split('/')[:-1])
config_path = path.join(model_dir, 'config.json')

cfg = Config.load_from_json(config_path)

# No beam search for validation
cfg.set('batch_size', 1)
cfg.set('train', False)
cfg.set('use_beam_search', False)
cfg.set('start_training_from_beginning', True)
cfg.set('model_path', model_path)

runner = Runner(cfg)

sample_txts = [line for line in open(sample_txts_path, 'r')]

with h5py.File(embs_out_path) as embs_file:
    embeddings = embs_file.create_dataset('embeddings', dtype='float32',
                                          shape=(len(sample_txts),
                                                 cfg.get('num_hidden_units')))

    for i, sample in tqdm(enumerate(sample_txts), total=len(sample_txts)):
        _, add_tensors = runner.inference(sample, additional_tensor_names=[EMB_TENSOR_NAME])
        input_emb = add_tensors[0].reshape(-1)[-cfg.get('num_hidden_units'):]
        embeddings[i] = input_emb

print('Finished generating internal embeddings for samples in %s' % sample_txts_path)
