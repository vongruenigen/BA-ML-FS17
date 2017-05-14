#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
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

    def __init__(self, cfg):
        '''Constructor of the DataLoader class. It only expects
           a Config object as the first and only parameter.'''
        self.cfg = cfg

    def load_conversations(self, path, vocabulary, disable_forwarding=False):
        '''Loads the conversations and returns a generator which
           yields one whole conversation with possibly multiple
           turns.'''
        all_convs = []
        curr_conv = []
        turn_flag = False

        forward_to = self.cfg.get('global_step')
        use_last_output_as_input = self.cfg.get('use_last_output_as_input')

        if not disable_forwarding and not self.cfg.get('start_training_from_beginning'):
            # We need to half the forward_to value in case
            # that outputs are used as new inputs
            if use_last_output_as_input:
                forward_to /= 2
                forward_to = int(forward_to)

            if forward_to > 0:
                logger.info('Forwarding to sample %i as indicated by global_step' % forward_to)
        else:
            forward_to = 0

        last_sentence = None

        while True:
            for i, line in enumerate(open(path, 'r', encoding='utf8')):
                # Skip as much samples as we've already seen indicated by the global_step
                if i < forward_to:
                    continue

                if line.strip() == self.SPLIT_CONV_SYM or len(curr_conv) >= self.cfg.get('batch_size'):
                    # TODO: How to fix the problem that a conversation of an odd
                    #       number of turns cannot be easily converted to a training sample?
                    if len(curr_conv) % 2 != 0:
                        del curr_conv[-1]

                    if len(curr_conv) > 0:
                        yield curr_conv

                    curr_conv = []

                    continue

                text_indices = self.convert_text_to_indices(line, vocabulary)

                # shorten the text in case it's longer than configured to be allowed to
                if len(text_indices) > self.cfg.get('max_input_length'):
                    text_indices = text_indices[:self.cfg.get('max_input_length')]

                if use_last_output_as_input and last_sentence is not None and len(curr_conv) > 1:
                    curr_conv.append(last_sentence)

                curr_conv.append(text_indices)
                last_sentence = text_indices

            if self.cfg.get('train'):
                logger.warn('WARNING: Went through all the data, starting from the beginning again!')
            else:
                raise StopIteration('finished the validation/test data')

    def get_tokenizer(self):
        '''Creates a tokenizer based on the configuration
           in the config object supplied in the constructor.
           The actual object is not returned but rather the
           tokenization function provided by the object.'''
        tokenizer_name = self.cfg.get('tokenizer_name')

        if tokenizer_name == 'word_tokenize':
            return nltk.word_tokenize
        elif tokenizer_name == 'none':
            return (lambda x: x.split(' '))
        else:
            raise Exception('unknown tokenizer %s set in configuration' % tokenizer_name)

    def convert_text_to_indices(self, line, vocabulary):
        '''Converts a text to the respective list of indices
           by using the vocabulary dictionary.'''
        line_parts = self.__preprocess_and_tokenize_line(line, vocabulary)
        line_parts = map(lambda w: vocabulary[w] if w in vocabulary else Config.UNKNOWN_WORD_IDX,
                         line_parts)
        line_parts = list(line_parts)
        line_parts.append(Config.EOS_WORD_IDX)

        if len(line_parts) < self.cfg.get('max_input_length'):
            max_input_length = self.cfg.get('max_input_length')
            padding_parts = [Config.PAD_WORD_IDX for i in range(max_input_length - len(line_parts) - 1)]
            line_parts += padding_parts

        return list(line_parts)

    def convert_indices_to_text(self, text_idxs, rev_vocabulary, trim_eos_pad=True):
        '''Converts a list of indices to the respective texts
           by using the vocabulary dictionary. Note that this
           function expects a reversed version of the vocabulary
           used to encode the text via convert_text_to_indices().'''
        skip_idxs = [Config.UNKNOWN_WORD_IDX, Config.PAD_WORD_IDX,
                     Config.EOS_WORD_IDX, Config.GO_WORD_IDX]
        text_idxs = list(text_idxs)

        # Remove anything after the first EOS token
        if Config.EOS_WORD_IDX in text_idxs and trim_eos_pad:
            text_idxs = text_idxs[:text_idxs.index(Config.EOS_WORD_IDX)]

        rev_text_idxs = list(reversed(text_idxs))
        shortened_idxs = []

        if trim_eos_pad:
            # Let's remove the padded <unknown> words before converting
            # the indices into text again.
            for idx in rev_text_idxs:
                if idx in skip_idxs and len(shortened_idxs) == 0:
                    continue
                else:
                    shortened_idxs.append(idx)
        else:
            shortened_idxs = rev_text_idxs

        if len(shortened_idxs) == 0:
            shortened_idxs = rev_text_idxs

        if len(shortened_idxs) > 0 and shortened_idxs[-1] == Config.GO_WORD_IDX:
            shortened_idxs = shortened_idxs[:-1]

        return ' '.join(map(lambda x: rev_vocabulary[x], reversed(shortened_idxs)))

    def __preprocess_and_tokenize_line(self, line, vocabulary):
        '''Preprocesses a given line (e.g. removes unwanted chars),
           tokenizes it and returns it.'''
        tknz = self.get_tokenizer()
        line = re.sub(self.WHITELIST_REGEX, '', line)

        line_parts = tknz(line)
        line_parts = map(lambda x: x.lower().strip(), line_parts)

        return list(line_parts)
