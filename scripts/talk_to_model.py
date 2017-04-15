#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Simple script which allows for interacting
#              with a trained model.
#

import sys
import os
import re
import helpers

from os import path

# Shut down annoying log messages from tensorflow ...
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)

helpers.expand_import_path_to_source()

from os import path
from data_loader import DataLoader
from runner import Runner
from config import Config

argv = sys.argv[1:]
results_dir = path.abspath(path.join(path.dirname(__file__), '..', 'results'))

available_models = []

for entry in os.listdir(results_dir):
    full_entry = path.join(results_dir, entry)
    find_model = lambda file: 'chkp' in file and 'data' in file

    if path.isdir(full_entry):
        found_models = list(filter(find_model, os.listdir(full_entry)))

        if any(found_models):
            found_models = [path.join(results_dir, full_entry, m) for m in found_models]
            available_models += found_models

if len(available_models) > 0:
    print('The following models are available, please specify which one you want to load:\n')

    for i, avm in enumerate(available_models):
        print('\t[%i] %s' % (i, '/'.join(avm.split('/')[-2:])))
else:
    print('No models available in results/, exiting!')
    sys.exit(2)

model_nr = int(input('\n(Model-Nr.)  > '))
selected_model = available_models[model_nr]
print('You have selected the model stored at %s' % selected_model)

config_path = path.join('/'.join(selected_model.split('/')[:-1]), 'config.json')
config = Config.load_from_json(config_path)

config.set('train', False) # == inference
config.set('batch_size', 1)
config.set('model_path', selected_model)

runner = Runner(config)

print('Welcome! You can now talk to the loaded model.')
print('(Simply enter a message and press enter, enter "exit" for quitting)')

ask = lambda: input('(Input)  > ')
msg = None

try:
    while True:
        msg = ask()

        if msg.lower() == 'exit':
            break

        ans = runner.inference(msg)
        print('(Answer) > %s' % ans)
except Exception as e:
    print('(Exiting conversation due to an error: %s)' % e)

print('Bye bye!')
