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

if len(argv) == 0:
    print('ERROR: Expected a config to load the model with')
    print('       (e.g. python scripts/talk_to_model.py <json-config>')
    sys.exit(2)

available_models = []

for entry in os.listdir(results_dir):
    full_entry = path.join(results_dir, entry)

    if path.isdir(full_entry) and any(map(lambda x: 'chkp' in x, os.listdir(full_entry))):
        available_models.append(entry)

if len(available_models) > 0:
    print('The following models are available, please specify which one you want to load:\n')

    for i, avm in enumerate(available_models):
        print('\t[%i] %s' % (i, avm.split('/')[-1]))
else:
    print('No models available in results/, exiting!')
    sys.exit(2)

model_nr = int(input('\n(Model-Nr.)  > '))
selected_model = available_models[model_nr]
print('You have selected the model stored at %s' % selected_model)

config_path = argv[0]
config = Config.load_from_json(config_path)
config.set('train', False) # == inference
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
    import pdb
    pdb.set_trace()
    print('(Exiting conversation due to an error)')

print('Bye bye!')