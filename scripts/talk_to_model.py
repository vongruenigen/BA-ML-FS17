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

# Shut down annoying log messages from tensorflow ...
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)

helpers.expand_import_path_to_source()

from os import path
from data_loader import DataLoader
from runner import Runner
from config import Config

argv = sys.argv[1:]

if len(argv) == 0:
    print('ERROR: Expected a config to load the model with')
    print('       (e.g. python scripts/talk_to_model.py <json-config>')
    sys.exit(2)

config_path = argv[0]
config = Config.load_from_json(config_path)
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