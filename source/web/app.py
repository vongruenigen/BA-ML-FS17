#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: This module is responsible for running a
#              smallish web application which can be used
#              to do online inference.
#

import sys
import os
from os import path

# Load source path
SOURCE_PATH = path.realpath(path.join(path.dirname(__file__), '..'))
sys.path.insert(0, SOURCE_PATH)

import flask
import json

from flask import request

from config import Config
from runner import Runner

template_dir = path.realpath(path.join(path.dirname(__file__), 'templates'))
static_dir = path.realpath(path.join(path.dirname(__file__), 'static'))

app = flask.Flask('BA-ML-FS17-online-inference',
                  template_folder=template_dir,
                  static_folder=static_dir)

RESULTS_DIRECTORY = path.realpath(path.join(path.dirname(__file__), '..', '..', 'results'))
REQUIRED_RESULT_FILES = ['config.json', 'checkpoint']

if not path.isdir(RESULTS_DIRECTORY):
    raise Exception('Results directory is missing, exiting (we checked "%s")' % RESULTS_DIRECTORY)

global current_model, current_runner

current_model = None
current_runner = None

@app.route('/')
def index():
    '''Returns the index page for the online inference app.'''
    return flask.render_template('index.html')

@app.route('/start_session/<model>', methods=['POST'])
def start_session(model):
    global current_runner, current_model

    if current_runner is not None:
        return 'A session is already running!', 403

    if model not in get_available_models():
        return 'The selected model does not exist!', 404

    current_model = model
    cfg_obj = get_config(path.join(RESULTS_DIRECTORY, model))
    current_runner = Runner(cfg_obj)

    # Run one sample before returning to ensure that the
    # graph and model are already loaded when the user starts
    # doing inference
    current_runner.inference('blub')

    return 'Model loaded', 200

@app.route('/get_session')
def get_session():
    global current_runner, current_model

    if current_runner is None:
        return 'No session started yet!', 404
    else:
        return current_model

@app.route('/run_inference', methods=['POST'])
def run_inference():
    global current_runner

    if current_runner is None:
        return 'No session started yet!', 403
    else:
        text = request.get_data().decode('utf-8')
        answ, _ = current_runner.inference(text)
        return answ, 200

@app.route('/stop_session', methods=['POST'])
def stop_session():
    global current_runner, current_model

    if current_runner is None:
        return 'No session started yet!', 403
    else:
        current_runner.close()
        current_runner = None
        current_model = None

@app.route('/get_models')
def get_models():
    return flask.jsonify(get_available_models()), 200

def expand_results_path(res_dir):
    '''Expands the given model name to the full results path.'''
    return path.join(RESULTS_DIRECTORY, res_dir)

def get_config(result_dir):
    full_path = path.join(result_dir, 'config.json')
    config_dict = {}

    try:
        with open(full_path, 'r') as f:
            config_dict = json.load(f)
    except:
        logger.error('Error while loading json config')

    config_dict['train'] = False
    config_dict['device'] = 'cpu:0'
    config_dict['model_path'] = result_dir

    return Config(config_dict)

def get_available_models():
    '''Returns the list of available models in results/.'''
    available_models = []

    for res_dir in os.listdir(RESULTS_DIRECTORY):
        full_path = expand_results_path(res_dir)

        if not path.isdir(full_path):
            continue

        files_in_dir = os.listdir(full_path)

        if all(map(lambda x: x in files_in_dir, REQUIRED_RESULT_FILES)):
            available_models.append(res_dir.strip('/'))

    return available_models

if __name__ == '__main__':
    app.run(port=9001)
