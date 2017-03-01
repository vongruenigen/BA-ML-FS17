#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the DataLoader class
#              which is responsible for loading the training
#              and testing data.
#

class DataLoader(object):
    '''This class is responsible for loading and preprocessing
       the training and test data used in this project.'''

    def __init__(self, cfg):
        '''Constructor of the DataLoader class. It only expects
           a Config object as the first and only parameter.'''
        self.cfg = cfg
