#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module is responsible for training
#              and running of a given model.
#

class Runner(object):
    '''This class is responsible for the training,
       testing and running of the model.'''

    def __initialize__(self, cfg_path):
        '''Constructor of the Runner class. It expects
           the path to the config file to runs as the
           only parameter.'''
        self.cfg_path = cfg_path
        self.cfg = Config.load_from_json(self.cfg_path)
