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

from multimodel.constants import DEFAULT_CONFIG_PARAMS

class Config(object):
    '''Config class for holding the configuration parameters.'''
    def __init__(self, cfg_obj={}):
        '''Creates a new Config object with the parameters
           from the cfg_obj parameter. The default parameters
           are used if no cfg_obj object is given.'''
        self.cfg_obj = DEFAULT_CONFIG_PARAMS.copy()
        self.__deep_merge(self.cfg_obj, cfg_obj)
        self.__create_id()

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
