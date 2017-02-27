#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the Config class
#              which is responsible for holding the
#              configuration while running the project.
#

import json

class Config(object):
    '''Config class for holding the configuration parameters. It
       provides dynamically generated properties for all params.'''

    # Contains the defaults which are applied in case any config
    # parameter is missing. ALL parameters should have an entry
    # in this list. If no meaningful value can be defined beforehand,
    # None should be used as the value.
    DEFAULT_PARAMS = {
        # Number of epochs to use while training
        'epochs': 100,

        # Batch size which will be used when training
        'batch_size': 100,

        # Number of layers (or cells) in the encoder/decoder
        # parts of the network
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,

        # Number of hidden units in each of the layers in each cell
        'num_hidden_units': 100,

        # Defines the cell type which will be used, either 'RNN', 'LSTM' or 'GRU'
        'cell_type': 'LSTM',

        # Defines the buckets which are used to simplify training
        'buckets': [
            [0, 10],
            [10, 20],
            [20, 30],
            [30, 40],
            [40, 50],
            [50, 100]
        ],

        # Defines the used vocabulary
        'vocabulary': None,

        # Defines the path to the word2vec embeddings to load (if used)
        'w2v_embeddings': None,

        # Defines the path to the fasttext embeddings to load (if used)
        'ft_embeddings': None,

        # Defines wether the used RNN processes the data in both directions
        # (backward/forward) or if it should be unidirectional.
        'bidirectional': False,

        # Defines wether the attention mechanism should be activated in the
        # decoder part of the sequence-to-sequence model.
        'use_attention': False,

        # Defines wether the debuggin environment should be activated.
        'debug': False,

        # Defines the optimizer and the hyperparameters to use
        'optimizer_name': 'AdaDelta',
        'optimizer_parameters': {}
    }

    def __init__(self, cfg_obj, ignore_unknown=True):
        '''Creates a new Config object with the parameters
           from the cfg_obj parameter.'''
        self.cfg_obj = self.DEFAULT_PARAMS.copy()
        self.cfg_obj.update(cfg_obj)

    def get(self, name):
        '''Returns the value for the given name stored in the current config object.'''
        if name in self.cfg_obj:
            return self.cfg_obj[name]
        else:
            raise Exception('there is no value for the name %s in the config' % name)

    def set(self, name, value):
        '''Sets the value for the given name in the current config object.'''
        if name in self.cfg_obj:
            self.cfg_obj[name] = value
        else:
            raise Exception('there is no value for the name %s in the config' % name)

    @staticmethod
    def load_from_json(json_path):
        '''Loads the JSON config from the given path and
           creates a Config object for holding the parameters.'''
        with open(json_path, 'r') as f:
            return Config(json.load(f))
