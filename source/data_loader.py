#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the DataLoader class
#              which is responsible for loading the training
#              and testing data.
#

import nltk
import re
import utils
import logger

class DataLoader(object):
    '''This class is responsible for loading and preprocessing
       the training and test data used in this project. This class
       can load data in the specified format: The conversations have
       to be fully expaned (e.g. by using the script preprocess_cornell_movie_dialogues_dataset.py).
       In the dialogue files, each dialogue should be written down one turn per line. This means that
       a dialogue where each participant says two sentences has four lines in complete. Each dialogue
       has to be finished by a special token (defined in the SPLIT_CONV_SYM constant).'''

    # NOTE: All datasets have to be preprocessed again if this symbol
    #       changes, otherwise the DataLoader class won't be able to
    #       load the conversations correctly!
    SPLIT_CONV_SYM = '<<<<<END-CONV>>>>>'

    # Regular expression which is used to filter unwanted characters
    WHITELIST_REGEX = '[^A-Za-z0-9\.,\?\!\s]+'

    # Index of the embedding for the unknown word in the embedding matrix.
    UNKNOWN_WORD_IDX = 0

    def __init__(self, cfg):
        '''Constructor of the DataLoader class. It only expects
           a Config object as the first and only parameter.'''
        self.cfg = cfg

    def load_conversations(self, path, vocabulary):
        '''Loads the conversations and returns a generator which
           yields one whole conversation with possibly multiple
           turns.'''
        all_convs = []
        curr_conv = []
        turn_flag = False

        while True:
            for i, line in enumerate(open(path, 'r')):
                if line.strip() == self.SPLIT_CONV_SYM:
                    # TODO: How to fix the problem that a conversation of an odd
                    #       number of turns cannot be easily converted to a training sample?
                    if len(curr_conv) % 2 != 0:
                        del curr_conv[-1]

                    yield curr_conv
                    curr_conv = []
                else:
                    curr_conv.append(self.__convert_line_to_indices(line, vocabulary))

            logger.warning('WARNING: Went through all the data, starting from the beginning again!')

    def __get_tokenizer(self):
        '''Creates a tokenizer based on the configuration
           in the config object supplied in the constructor.
           The actual object is not returned but rather the
           tokenization function provided by the object.'''
        tokenizer_name = self.cfg.get('tokenizer_name')

        if tokenizer_name == 'word_tokenize':
            return nltk.word_tokenize
        else:
            raise Exception('unknown tokenizer %s set in configuration' % tokenizer_name)

    def __convert_line_to_indices(self, line, vocabulary):
        '''Parses a single line of a conversation and returns
           it as a list of indices.'''
        line_parts = self.__preprocess_and_tokenize_line(line, vocabulary)
        line_parts = map(lambda w: vocabulary[w] if w in vocabulary else self.UNKNOWN_WORD_IDX,
                         line_parts)

        # reverse the input in case it's configured
        if self.cfg.get('reverse_input'):
            line_parts = reversed(line_parts)

        return list(line_parts)

    def __preprocess_and_tokenize_line(self, line, vocabulary):
        '''Preprocesses a given line (e.g. removes unwanted chars),
           tokenizes it and returns it.'''
        tknz = self.__get_tokenizer()
        line = re.sub(self.WHITELIST_REGEX, '', line)

        line_parts = tknz(line)
        line_parts = map(lambda x: x.lower().strip(), line_parts)

        return list(line_parts)
