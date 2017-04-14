#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: This module contains helper functions which
#              are used in the scripts.
#

import sys

from os import path

SOURCE_PATH = path.realpath(path.join(path.dirname(__file__), '..', 'source'))

def expand_import_path_to_source():
    '''This method ensures that, when called, we can import modules
       from the source/ directory when using a script.'''
    if SOURCE_PATH not in sys.path:
        sys.path.insert(0, SOURCE_PATH)
