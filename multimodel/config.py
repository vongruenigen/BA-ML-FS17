#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the Config class
#              which is responsible for holding the
#              configuration while running the project.
#

import json
import time
import collections

from os import path

class Config(object):
    '''Config class for holding the configuration parameters.'''

    # Path constants
    ROOT_PATH    = path.realpath(path.join(path.dirname(__file__), '..'))
    LOGS_PATH    = path.join(ROOT_PATH, 'logs')
    DATA_PATH    = path.join(ROOT_PATH, 'data')
    CONFIG_PATH  = path.join(ROOT_PATH, 'configs')
    RESULTS_PATH = path.join(ROOT_PATH, 'results')

    # Vocabulary constants
    UNKNOWN_WORD_IDX   = 0
    UNKNOWN_WORD_TOKEN = '<unknown>'
    PAD_WORD_IDX       = 1
    PAD_WORD_TOKEN     = '<pad>'
    EOS_WORD_IDX       = 2
    EOS_WORD_TOKEN     = '<eos>'

    # Contains the defaults which are applied in case any config
    # parameter is missing. ALL parameters should have an entry
    # in this list. If no meaningful value can be defined beforehand,
    # None should be used as the value.
    DEFAULT_PARAMS = {
        # Id of the experiment (autogenerated)
        'id': None,

        # Signifies the git version used when running the experiment
        'git_rev': None,

        # Random seed used to initialize numpy and tensorflow rngs
        'random_seed': 1337,

        # Path to the model to load (optional). The path MUST point
        # to a *.chkp file, not any of the related files (*.index, *.meta, etc.).
        # This means that the path might be "directory/model-1.chkp-1001" even though
        # this file does not exist on the hard drive.
        'model_path': None,

        # Name of the experiment (optional)
        'name': None,

        # Number of epochs to do while training
        'epochs': 1,

        # Batch size which will be used when training
        'batch_size': 4,

        # Defines how much batches will be considered per epoch.
        'batches_per_epoch': 1000,

        # Defines wether we should train on actual sequence learning or rather
        # just learn to copy the input sequences to the output. This flag can
        # be used to debug the model and see if it still works as expected.
        'train_on_copy': False,

        # Number of layers (or cells) in the encoder/decoder
        # parts of the network
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,

        # Number of hidden units in each of the layers in each cell
        'num_hidden_units': 1000,

        # Defines the cell type which will be used, either 'RNN', 'LSTM' or 'GRU'
        'cell_type': 'LSTM',

        # Defines the used vocabulary. This has to be a pickle file containing
        # a dictionary which maps words to indices in the embeddings matrix.
        'vocabulary': None,

        # Defines wether there should be examples printed to the console
        # of the input and respective output of the model as clear text.
        'show_predictions_while_training': False,

        # Defines wether a list of indices or real text should be shown
        # when showing the predictions while training.
        'show_text_when_showing_predictions': True,

        # Defines the path to the word2vec embeddings to load (if used)
        'w2v_embeddings': None,

        # Defines the path to the fasttext embeddings to load (if used)
        'ft_embeddings': None,

        # The embeddings can be set in the config to pass it to the model.
        # However, it cannot be set via the JSON config and is only availabe
        # after loading the embeddings.
        'embeddings': None,

        # The vocabulary will be set when the embeddings are loaded.
        'vocabulary': None, 

        # Defines the vocabulary to use. It should be a path to a pickle
        # file containing the vocabulary as a dict where the keys are the
        # words and the values the indices of the respecting word in the
        # embedding matrix.
        'vocabulary': None,

        # Defines wether the used RNN processes the data in both directions
        # (backward/forward) or if it should be unidirectional.
        'bidirectional': False,

        # Defines wether the attention mechanism should be activated in the
        # decoder part of the sequence-to-sequence model.
        'use_attention': False,

        # Defines wether the input sample should be reversed before it's
        # fed to the encoder. This is a trick used in the original seq2seq
        # paper.
        'reverse_input': False,

        # Defines wether the dropout mechanism should be used to mitigate
        # the problem of overfitting.
        'use_dropout': False,

        # Defines wether the debuggin environment should be activated.
        'debug': False,

        # Defines how much words should be considered in the vocabulary
        # in case no embeddings are provided.
        'max_vocabulary_size': 10000,

        # Defines how much words should be samples when using samples softmax.
        # The sampled softmax is only used if the number of words in the vocabulary
        # exceeds the value in 'max_vocabulary_size'.
        'sampled_softmax_number_of_samples': 10,

        # Defines how much dimensions should be used when using random embeddings.
        'max_random_embeddings_size': 10,

        # Defines the optimizer and the hyperparameters to use
        'optimizer_name': 'AdaDelta',
        'optimizer_parameters': {},

        # Defines how much checkpoints should be kept at max. Defaults to 5.
        'checkpoint_max_to_keep': 5,

        # Defines how much of the input neurons should still be considered
        # when applying the dropout mechanism. The value 1.0 means that all
        # neurons are used and none is dropped.
        'dropout_input_keep': 1.0,

        # Defines how much of the output neurons should still be considered
        # when applying the dropout mechanism. The value 1.0 means that all
        # neurons are used and none is dropped.
        'dropout_output_keep': 1.0,

        # Defines which tokenizer should be used to parse the conversational
        # texts. The default tokenizers is the 'word_tokenizer' of the ntlk
        # module.
        'tokenizer_name': 'word_tokenize',

        # Defines which training data to load. It tries to load the conversation
        # of the file at the given path, this file must obey the correct format
        # described in the DataLoader class.
        'training_data': None,

        # Defines which test data to load. It tries to load the conversation
        # of the file at the given path, this file must obey the correct format
        # described in the DataLoader class.
        'test_data': None,

        # Defines the maximum allowed length of the input sentences. Longer inputs
        # will be reduced to this maximum.
        'max_input_length': 100,

        # Defines the standard deviation which should be used when randomly applying
        # time-dependent gradient noise as seen in: https://arxiv.org/pdf/1511.06807.pdf.
        # Defaults to zero, which is the same as not applying any noise.
        'gradient_noise_stddev': 0.0,

        # Defines the max values the gradients will be clipped to if neccessary.
        'gradient_clipping_norm': 10.0,

        # Part to specifiy and configurate the model
        'model_name': 'seq2seq',
        'model_config': {}
    }

    def __init__(self, cfg_obj={}):
        '''Creates a new Config object with the parameters
           from the cfg_obj parameter. The default parameters
           are used if no cfg_obj object is given.'''
        self.cfg_obj = self.DEFAULT_PARAMS.copy()
        self.__deep_merge(self.cfg_obj, cfg_obj)
        self.__create_id()

    @staticmethod
    def get_model_ctor(name):
        '''Returns the constructor for the model with the given name.'''
        if name not in self.MODEL_CTOR_MAPPING:
            raise Exception('there is no model with the name "%s"\n'
                            'the following models are available: '
                            ', '.join(list(self.MODEL_CTOR_MAPPING.keys())))

        return self.MODEL_CTOR_MAPPING[name]

    def get(self, names):
        '''Returns the value for the given names stored in the current config object.
           Multiple names separated with a slash (e.g. model/value) can be entered for
           accessing deeper config values (e.g. specific model params). '''
        names = names.split('/')

        def read_val(name):
            if name in self.cfg_obj:
                return self.cfg_obj[name]
            else:
                raise Exception('there is no value for the name %s in the config' % name)

        val = self.cfg_obj

        for n in names:
            if n not in val:
                raise Exception('config value %s does not exist' % names)

            val = val.get(n)

        return val

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

    def __create_id(self):
        '''This method is responsible for creating a unique id
           for the loaded configuration. This id is then used to
           create the results directory.'''
        name = self.get('name')
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

        if name:
            timestamp = '%s_%s' % (timestamp, name)

        self.set('id', timestamp)

    def __deep_merge(self, target, to_merge):
        '''Deep merges the given dictionaries.'''
        for k, v in to_merge.items():
            if k in target and isinstance(target[k], dict):
                dict_merge(target[k], to_merge[k])
            else:
                target[k] = to_merge[k]
