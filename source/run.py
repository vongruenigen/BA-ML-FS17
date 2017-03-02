#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This script is responsible for starting
#              either an interactive session using an
#              already trained model or to train a model
#              from scratch.
#

import json
import sys
import os
import numpy as np
import time

from config import Config
from os import path
from subprocess import Popen, PIPE

#
# Parameters for the run
#
vocabulary_path = ''
embeddings_path = ''
test_data = ''
validation_data_path = ''
verbose = False
git_sha = ''
random_seed = int(time.time())

#
# Argument handling
#
argv = sys.argv[1:]

if len(argv) == 0 or argv[0] == '':
    print("ERROR: JSON config is missing")
    print("       (e.g. ./run.sh config.json)")
    sys.exit(1)

# Try to load the SHA of the current git revision or error out
git_proc = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, stderr=PIPE)

git_out = git_proc.communicate()[0][0:-1] # cut off \n and second line
git_rev = git_out.decode('utf-8')
git_err = git_proc.returncode

if git_err != 0:
    print('ERROR: error while fetching current git revision SHA')
    sys.exit(1)

for i, arg in enumerate(argv):
    config_path = argv[i]
    configs = []

    if path.isdir(config_path):
        for c in os.listdir(config_path):
            if c.endswith('.json'):
                configs.append(path.join(config_path, c))
    else:
        configs = [config_path]

    print('The following configs will be run:\n* %s' % '\n* '.join(configs))

    for cfg in configs:
        print('Starting run with file %s' % cfg)

        with open(cfg) as f:
            cfg_obj = json.loads(f.read())

        # Take the config name for the results directory
        # in case of no name is defined
        if not 'name' in cfg_obj:
            cfg_obj['name'] = path.splitext(path.basename(cfg))[0]

        # Store the git hash in the config
        cfg_obj['git_rev'] = git_rev

        if 'random_seed' in cfg_obj:
            random_seed = cfg_obj['random_seed']
            print('Using configured seed: %s' % str(random_seed))
        elif 'random_seed' in os.environ:
            random_seed = int(os.environ['random_seed'])
            cfg_obj['random_seed'] = random_seed
            print('Using injected seed: %s' % str(random_seed))
        else:
            cfg_obj['random_seed'] = random_seed
            print('Using unix timestamp as seed: %s' % random_seed)

        np.random.seed(random_seed)

        from runner import Runner

        #
        # Execute the run!
        #
        config = Config(cfg_obj)
        runner = Runner(config)
        runner.train()

        print('Finished run with file %s' % config_path)
