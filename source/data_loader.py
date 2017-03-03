#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the DataLoader class
#              which is responsible for loading the training
#              and testing data.
#

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
    #       load the conversations!
    SPLIT_CONV_SYM = '<<<<<END-CONV>>>>>'

    def __init__(self, cfg):
        '''Constructor of the DataLoader class. It only expects
           a Config object as the first and only parameter.'''
        self.cfg = cfg
