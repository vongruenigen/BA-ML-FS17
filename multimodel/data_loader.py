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

from config import Config

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
                    text_indices = self.convert_text_to_indices(line, vocabulary)

                    # reverse the input in case it's configured
                    if self.cfg.get('reverse_input'):
                        text_indices = reversed(text_indices)

                    # shorten the text in case it's longer than configured to be allowed to
                    if len(text_indices) > self.cfg.get('max_input_length'):
                        text_indices = text_indices[:self.cfg.get('max_input_length')]

                    curr_conv.append(text_indices)

            logger.warn('WARNING: Went through all the data, starting from the beginning again!')

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

    def convert_text_to_indices(self, line, vocabulary):
        '''Converts a text to the respective list of indices
           by using the vocabulary dictionary.'''

        # we need to pad the lines in case we're dealing with a memn2n model
        if self.cfg.get('model_name'):
            line = line.lpad(self.cfg.get('memn2n/sentence_length'))

        line_parts = self.__preprocess_and_tokenize_line(line, vocabulary)
        line_parts = map(lambda w: vocabulary[w] if w in vocabulary else Config.UNKNOWN_WORD_IDX,
                         line_parts)

        return list(line_parts)

    def convert_indices_to_text(self, text_idxs, rev_vocabulary):
        '''Converts a list of indices to the respective texts
           by using the vocabulary dictionary. Note that this
           function expects a reversed version of the vocabulary
           used to encode the text via convert_text_to_indices().'''
        skip_idxs = [Config.PAD_WORD_IDX]
        shortened_idxs = []

        # Let's remove the padded <unknown> words before converting
        # the indices into text again.
        for idx in reversed(text_idxs):
            if idx in skip_idxs:
                continue
            else:
                shortened_idxs.append(idx)

        return ' '.join(map(lambda x: str(rev_vocabulary[x]), reversed(shortened_idxs)))

    def __preprocess_and_tokenize_line(self, line, vocabulary):
        '''Preprocesses a given line (e.g. removes unwanted chars),
           tokenizes it and returns it.'''
        tknz = self.__get_tokenizer()
        line = re.sub(self.WHITELIST_REGEX, '', line)

        line_parts = tknz(line)
        line_parts = map(lambda x: x.lower().strip(), line_parts)

        return list(line_parts)
