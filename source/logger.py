#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: Simple wrapper for the python logging library.
#

import sys
import logging
import tensorflow as tf

from os import path
from config import Config

LOGGER_NAME = 'BA-ML-FS17'
LOGGER_FORMAT = '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
LOGFILE_PATH = path.join(Config.LOGS_PATH, 'ba-ml-fs17.log')
TF_LOGFILE_PATH = path.join(Config.LOGS_PATH, 'tf.log')

mod = sys.modules[__name__]
mod.logger = None

def log(lvl, msg):
    if mod.logger is None:
        raise Exception('init_logger() must be called one time before any logging can be done')
    mod.logger.log(lvl, msg)
 
def info(msg):
    log(logging.INFO, msg)

def warn(msg):
    log(logging.WARN, msg)

def error(msg):
    log(logging.ERROR, msg)

def fatal(msg, exit_code=2):
    print('--- !!! FATAL ERROR !!! ---')
    print(msg)
    sys.exit(exit_code)

def init_logger(cfg):
    level = logging.DEBUG if cfg.get('debug') else logging.INFO
    formatter = logging.Formatter(LOGGER_FORMAT)

    mod.logger = logging.getLogger(LOGGER_NAME)
    mod.logger.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(LOGFILE_PATH)
    tf_file_handler = logging.FileHandler(TF_LOGFILE_PATH)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    tf_file_handler.setFormatter(formatter)

    mod.logger.addHandler(file_handler)
    mod.logger.addHandler(stream_handler)

    tf.logging._logger.addHandler(tf_file_handler)
