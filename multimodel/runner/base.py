#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the Base class used for the implementation
#              of different runners. Currently it's only used in tensorflow and
#              the keras runner.
#

import json
import os
import math
import numpy as np

from os import path

from multimodel import logger, utils
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
        self.load_embeddings_and_vocabulary()

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

    def load_embeddings_and_vocabulary(self):
        '''Loads the embeddings with the associated vocabulary
           and saves them for later usage in the DataLoader and
           while training/testing.'''
        if self.cfg.get('vocabulary') is None and self.cfg.get('use_integer_vocabulary'):
            # Build a stub dictionary which maps integers to integers for the defined range
            self.vocabulary = {str(x): x for x in range(self.cfg.get('max_vocabulary_size'))}
        else:
            self.vocabulary = utils.load_vocabulary(self.cfg.get('vocabulary'))

        embeddings_matrix = None

        if self.cfg.get('w2v_embeddings'):
            embeddings_matrix = utils.load_w2v_embeddings(self.cfg.get('w2v_embeddings'))
        elif self.cfg.get('ft_embeddings'):
            embeddings_matrix = utils.load_ft_embeddings(self.cfg.get('ft_embeddings'))
        elif self.cfg.get('use_random_embeddings'):
            max_idx = self.cfg.get('max_vocabulary_size')
            new_vocabulary = {}

            for k, v in self.vocabulary.items():
                if v <= max_idx+1:
                    new_vocabulary[k] = v

            self.vocabulary = new_vocabulary

            # uniform(-sqrt(3), sqrt(3)) has variance=1
            sqrt3 = math.sqrt(3)
            embeddings_matrix = np.random.uniform(-sqrt3, sqrt3, size=(len(self.vocabulary), 
                                                  self.cfg.get('max_random_embeddings_size')))

        if embeddings_matrix is not None:
            # Prepare the vocabulary and embeddings (e.g. add embedding for unknown words)
            embeddingx_matrix, self.vocabulary = utils.prepare_embeddings_and_vocabulary(
                                                            embeddings_matrix, self.vocabulary)

        self.embeddings = embeddings_matrix

        # Store the embeddings and vocabulary in the 
        self.cfg.set('embeddings', self.embeddings)
        self.cfg.set('vocabulary', self.vocabulary)

        # revert the vocabulary for the idx -> text usages
        self.rev_vocabulary = utils.reverse_vocabulary(self.vocabulary)

    def prepare_data_batch(self, all_data):
        '''Returns two lists, each of the size of the configured batch size. The first contains
           the input sentences (sentences which the first "person" said), the latter contains the
           list of expected answers.'''
        data_batch_x, data_batch_y = [], []
        batch_size = self.cfg.get('batch_size')

        conv_turn_idx = 0
        conversation = next(all_data)
        first_conv_turn = None
        second_conv_turn = None

        while len(data_batch_y) < batch_size and len(data_batch_x) < batch_size:
            # Check if we've reached the end of the conversation, in this
            # case we've to load the next conversation.
            try:
                if conv_turn_idx + 1 >= len(conversation):
                    conversation = next(all_data)
                    conv_turn_idx = 0
            except StopIteration as e:
                break # exit the loop in case there is no more data  

            first_conv_turn = conversation[conv_turn_idx]
            second_conv_turn = conversation[conv_turn_idx+1]
            
            if self.cfg.get('train_on_copy'):
                data_batch_x.append(first_conv_turn)
                data_batch_y.append(first_conv_turn)
            else:
                data_batch_x.append(first_conv_turn)
                data_batch_y.append(second_conv_turn)

            conv_turn_idx += 2

        return data_batch_x, data_batch_y
