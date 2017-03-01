#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module is responsible for training
#              and running of a given model.
#

import tensorflow as tf
import numpy as np
import contextlib
import utils
import os
import json

from model import Model
from config import Config
from os import path

class Runner(object):
    '''This class is responsible for the training,
       testing and running of the model.'''

    def __init__(self, cfg_path):
        '''Constructor of the Runner class. It expects
           the path to the config file to runs as the
           only parameter.'''
        if isinstance(cfg_path, str):
            self.cfg_path = cfg_path
            self.cfg = Config.load_from_json(self.cfg_path)
        elif isinstance(cfg_path, Config):
            self.cfg_path = None
            self.cfg = cfg_path
        else:
            raise Exception('cfg_path must be either a path or a Config object')

        self.__prepare_results_directory()

    def train(self):
        '''This method is responsible for training a model
           with the settings defined in the config.'''
        with self.__with_model() as (session, model):
            
            # TODO: Use real data!
            length_from=3
            length_to=8
            vocab_lower=2
            vocab_upper=10
            batch_size=100
            max_batches=5000
            batches_in_epoch=1000
            debug=True
            batches = utils.random_sequences(length_from=length_from, length_to=length_to,
                                             vocab_lower=vocab_lower, vocab_upper=vocab_upper,
                                             batch_size=batch_size)

            sum_losses = 0.0
            sum_iters = 0
            loss_track = []
            perplexity_track = []

            try:
                for epoch in range(self.cfg.get('epochs')):
                    for batch in range(max_batches+1):
                        batch_data = next(batches)
                        fd = model.make_train_inputs(batch_data, batch_data)
                        
                        _, l = session.run([model.train_op, model.loss], fd)

                        sum_losses += l
                        sum_iters += 1
                        
                        loss_track.append(l)
                        perplexity_track.append(np.exp(sum_losses / sum_iters))

                        if batch == 0 or batch % batches_in_epoch == 0:
                            print('batch {}'.format(batch))
                            print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
                            for i, (e_in, dt_pred) in enumerate(zip(
                                    fd[model.encoder_inputs].T,
                                    session.run(model.decoder_prediction_train, fd).T
                                )):
                                print('  sample {}:'.format(i + 1))
                                print('    enc input           > {}'.format(e_in))
                                print('    dec train predicted > {}'.format(dt_pred))
                                if i >= 2:
                                    break
                            print()
            except KeyboardInterrupt:
                print('training interrupted')
                # TODO: Save state when the training is interrupted?

            self.__store_metrics(loss_track, perplexity_track)

    def test(self):
        '''This method is responsible for evaluating a trained
           model with the settings defined in the config.'''
        with self.__with_model() as model:
            pass

    @contextlib.contextmanager
    def __with_model(self):
        '''This method is responsible for setting up the device,
           session and model which can then be used for training
           or inference.'''
        with tf.device(self.__get_device()):
            with self.__with_tf_session() as session:
                model = Model(self.cfg)

                # We need to initialize all the stuff setup when creating
                # the model instance. Otherwise we might get nasty error
                # messages regarding uninitialized variables.
                session.run(tf.global_variables_initializer())

                yield (session, model)

    @contextlib.contextmanager
    def __with_tf_session(self):
        '''This method is responsible for wrapping the execution
           of a given block within a tensorflow session.'''
        with tf.Graph().as_default():
            with tf.Session() as session:
                yield session

    @contextlib.contextmanager
    def __with_tf_saver(self, session):
        '''This method is responsible for ensuring that the state
           of the model is saved at all times using the tf.Saver
           class''' 
        pass

    def __get_device(self):
        '''Returns the name of the device to use when executing
           a computation.'''
        return '/gpu:0'

    def __store_metrics(self, loss_track, perplexity_track):
        '''This method is responsible for storing the metrics
           collected while training the model. Currently, this
           only includes the losses and perplexities.'''
        metrics_path = path.join(self.curr_exp_path, 'metrics.json')
        
        conv_to_float = lambda x: float(x)
        perplexity_track = list(map(conv_to_float, perplexity_track))
        loss_track = list(map(conv_to_float, loss_track))

        with open(metrics_path, 'w+') as f:
            json.dump({'loss': loss_track, 'perplexity': perplexity_track},
                      f, indent=4, sort_keys=True)

    def __prepare_results_directory(self):
        '''This method is responsible for preparing the results
           directory for the experiment with loaded config.'''
        if not path.isdir(Config.RESULTS_PATH):
            os.mkdir(Config.RESULTS_PATH)

        self.curr_exp_path = path.join(Config.RESULTS_PATH, self.cfg.get('id'))

        if path.isdir(self.curr_exp_path):
            raise Exception('A results directory with the name' +
                            ' %s already exists' % str(self.cfg.id))
        else:
            os.mkdir(self.curr_exp_path)

    def __get_model_path(self, version=0):
        '''Returns the path to store the model at as a string. An
           optional version can be specified and will be appended
           to the name of the stored file. If a model_path is set
           in the config, this will be returned and version will be
           ignored.'''
        if self.cfg.model_path:
            return self.cfg.model_path

        if not self.curr_exp_path:
            raise Exception('__prepare_results_directory() must be called before using __get_model_path()')

        return path.join(self.curr_exp_path, 'model-%s.ckpt' % str(version))
