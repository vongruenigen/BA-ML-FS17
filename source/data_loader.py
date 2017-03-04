#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the DataLoader class
#              which is responsible for loading the training
#              and testing data.
#

import nltk

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

    # Index of the embedding for the unknown word in the embedding matrix.
    UNKNOWN_WORD_IDX = 0

    def __init__(self, cfg):
        '''Constructor of the DataLoader class. It only expects
           a Config object as the first and only parameter.'''
        self.cfg = cfg

    def load_conversations(self, path, vocabulary):
        '''Loads the conversations at the given path.
           The files containing the conversations have
           to obey the correct format described in the
           module description. As a "unknown" word token,
           we use the 0.'''
        all_convs = []
        curr_conv = []
        turn_flag = False

        for i, line in enumerate(open(path, 'r')):
            if line == self.SPLIT_CONV_SYM:
                all_convs.append(curr_conv)
                curr_conv = []
            else:
                import pdb
                pdb.set_trace()
                curr_conv.append(self.__convert_line_to_indices(line))

        return all_convs

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
        tknz = self.__get_tokenizer()

        def get_word_idx(w):
            '''Helper function to map words to indices.'''
            if w in vocabulary:
                return vocabulary[w]
            else:
                return self.UNKNOWN_WORD_IDX

        line_parts = tknz(line)
        line_parts = map(lambda x: x.lower().strip(), line_parts)
        line_parts = map(get_word_idx, line_parts)

        return list(line_parts)

    def __load_conversations(self, path, vocabulary):
        '''Loads the conversations at the given path.'''
        convs = []

        with open(path, 'r') as f:
            pass
