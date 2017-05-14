#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#

import sys
import os
import json
import helpers
import numpy as np

helpers.expand_import_path_to_source()

from os import path
from runner import Runner
from config import Config
from data_loader import DataLoader

argv = sys.argv[1:]

if len(argv) < 3:
    print('ERROR: Missing mandatory argument!')
    print('       (python scripts/evaluate_trained_model.py <model-path> <test-corpus> <metrics-file> [<max-samples>]')
    sys.exit(2)

model_path = argv[0]
test_corpus_path = argv[1]
metrics_path = argv[2]
max_samples = None

if len(argv) > 3:
    max_samples = int(argv[3])

if not path.isfile(test_corpus_path):
    print('ERROR: The test-corpus param has to point to a file')
    sys.exit(2)

model_dir = '/'.join(model_path.split('/')[:-1])
config_path = path.join(model_dir, 'config.json')

cfg = Config.load_from_json(config_path)

# No beam search for validation
cfg.set('train', False)
cfg.set('use_beam_search', False)
cfg.set('start_training_from_beginning', True)
cfg.set('model_path', model_path)

runner = Runner(cfg)
test_loss, test_perplexity = runner.test(test_corpus_path, max_samples=max_samples)

print('=======\nResults\n=======')
print('Loss = %.5f' % test_loss)
print('Perplexity = %.5f' % test_perplexity)

with open(metrics_path, 'w+') as f:
    json.dump({'loss': test_loss, 'perplexity': test_perplexity}, f)
