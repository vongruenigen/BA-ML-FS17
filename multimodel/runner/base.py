#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the Base class used for the implementation
#              of different runners. Currently it's only used in tensorflow and
#              the keras runner.
#

import json
import os

from os import path

from multimodel import logger
from multimodel.constants import RESULTS_PATH
from multimodel.config import Config
from multimodel.data_loader import conversational

class Base(object):
    '''This class acts as a base for all runner implementations'''

    DATA_LOADER_MAP = {
        'conversational': conversational.ConversationalDataLoader
    }

    def __init__(self, cfg_path):
        '''Constructor of the Runner class. It expects the path to the config
           file to runs as the only parameter.'''
        if isinstance(cfg_path, str):
            self.cfg_path = cfg_path
            self.cfg = Config.load_from_json(self.cfg_path)
        elif isinstance(cfg_path, Config):
            self.cfg_path = None
            self.cfg = cfg_path
        else:
            raise Exception('cfg_path must be either a path or a Config object')

        logger.debug('The following config will be used')
        logger.debug(json.dumps(self.cfg.cfg_obj, indent=4, sort_keys=True))

        self.data_loader = self.get_data_loader()
        self.session = None
        self.model = None

        self.prepare_results_directory()

    def get_data_loader(self):
        '''Returns an instance of the configured data loader.'''
        return self.DATA_LOADER_MAP[self.cfg.get('data_loader')]

    def store_metrics(self, metrics_track):
        '''This method is responsible for storing the metrics
           collected while training the model. Currently, this
           only includes the losses and perplexities.'''
        metrics_path = path.join(self.curr_exp_path, 'metrics.json')
        metrics_obj = {}

        if len(metrics_track) > 0:
            for k in metrics_track[0].keys():
                metrics_obj[k] = []

            for metrics in metrics_track:
                for k in metrics_obj.keys():
                    metrics_obj[k].append(float(metrics[k]))

        with open(metrics_path, 'w+') as f:
            json.dump(metrics_obj, f, indent=4, sort_keys=True)

    def get_model_path(self, version=0):
        '''Returns the path to store the model at as a string. An
           optional version can be specified and will be appended
           to the name of the stored file. If a model_path is set
           in the config, this will be returned and version will be
           ignored.'''
        if self.cfg.get('model_path'):
            return self.cfg.get('model_path')

        if not self.curr_exp_path:
            raise Exception('prepare_results_directory() must be called before using get_model_path()')

        return path.join(self.curr_exp_path, 'model-%s' % str(version))

    def prepare_results_directory(self):
        '''This method is responsible for preparing the results
           directory for the experiment with loaded config.'''
        if not path.isdir(RESULTS_PATH):
            os.mkdir(RESULTS_PATH)

        self.curr_exp_path = path.join(RESULTS_PATH, self.cfg.get('id'))

        if path.isdir(self.curr_exp_path):
            raise Exception('A results directory with the name' +
                            ' %s already exists' % str(self.cfg.id))
        else:
            os.mkdir(self.curr_exp_path)
