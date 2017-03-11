#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Simple wrapper for the python logging library.
#

import sys
import logging

from os import path
from config import Config

LOGGER_NAME = 'multimodel'
LOGGER_FORMAT = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
LOGFILE_PATH = path.join(Config.LOGS_PATH, 'multimodel.log')

mod = sys.modules[__name__]
mod.logger = None

def log(lvl, msg):
    '''Logs the given message with the given level.'''
    if mod.logger is None:
        raise Exception('init_logger() must be called one time before any logging can be done')

    mod.logger.log(lvl, msg)
 
def info(msg):
    '''Logs the given message with the level info.'''
    log(logging.INFO, msg)

def warn(msg):
    '''Logs the given message with the level warn.'''
    log(logging.WARN, msg)

def error(msg):
    '''Logs the given message with the level error.'''
    log(logging.ERROR, msg)

def debug(msg):
    '''Logs the given message with the level debug.'''
    log(logging.DEBUG, msg)

def fatal(msg, exit_code=2):
    '''Logs the given message and terminates the program
       and returns the given exit code.'''
    error('--- !!! FATAL ERROR !!! ---')
    error(msg)
    sys.exit(exit_code)

def init_logger(cfg):
    '''Initializes the logger. This function has to be called
       before using any of the other logging functions.'''
    level = logging.DEBUG if cfg.get('debug') else logging.INFO
    formatter = logging.Formatter(LOGGER_FORMAT)

    mod.logger = logging.getLogger(LOGGER_NAME)
    mod.logger.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(LOGFILE_PATH)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    mod.logger.addHandler(file_handler)
    mod.logger.addHandler(stream_handler)
