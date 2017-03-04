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
import sys

from model import Model
from config import Config
from os import path
from data_loader import DataLoader

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

        self.__prepare_summary_writers()
        self.__prepare_results_directory()
        self.__load_embeddings()

        self.data_loader = DataLoader(self.cfg)

    def train(self):
        '''This method is responsible for training a model
           with the settings defined in the config.'''
        with self.__with_model() as (session, model):
            self.__setup_saver_and_restore_model(session)

            training_batches = []
            test_batches = []

            if self.cfg.get('training_data'):
                training_batches = self.data_loader.load_conversations(
                    self.cfg.get('training_data'),
                    self.vocabulary
                )

            if self.cfg.get('test_data'):
                test_batches = self.data_loader.load_conversations(
                    self.cfg.get('test_data'),
                    self.vocabulary
                )

            sum_losses = 0.0
            sum_iters = 0
            batches_per_epoch = self.cfg.get('batches_per_epoch')
            loss_track = []
            perplexity_track = []

            try:
                for epoch in range(self.cfg.get('epochs')):
                    for batch in range(batches_per_epoch+1):
                        batch_data_x, batch_data_y = self.__prepare_data_batch(training_batches)
                        import pdb
                        pdb.set_trace()
                        fd = model.make_train_inputs(batch_data_x, batch_data_y)

                        _, l = session.run([model.train_op, model.loss], fd)

                        sum_losses += l
                        sum_iters += 1
                        
                        loss_track.append(l)
                        perplexity_track.append(np.exp(sum_losses / sum_iters))

                        if batch == 0 or batch % batches_per_epoch == 0:
                            print('batch {}'.format(batch))
                            print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
                            # for i, (e_in, dt_pred) in enumerate(zip(
                            #         fd[model.encoder_inputs].T,
                            #         session.run(model.decoder_prediction_train, fd).T
                            #     )):
                            #     print('  sample {}:'.format(i + 1))
                            #     print('    enc input           > {}'.format(e_in))
                            #     print('    dec train predicted > {}'.format(dt_pred))
                            #     if i >= 2:
                            #         break
                            # print()

                            # Store model at certain checkpoints
                            self.__store_model(session, model)
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

                # Set the embeddings after creating a new instance of the model...
                model.set_embeddings(self.embeddings)

                # ..and build it afterwards
                model.build()

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

    def __store_model(self, session, model):
        '''Stores the given model in the current results directory.
           This model can then later be reloaded with the __load_model()
           method.'''
        model_path = self.__get_model_path()
        self.saver.save(session, model_path, global_step=model.get_global_step())
        print('Current version of the model stored at %s' % model_path)

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

    def __setup_saver_and_restore_model(self, session):
        '''Sets up the Saver which is used to store the model state after
           training. It also loads a previous model if referenced in the
           current configuration.'''
        model_path = self.cfg.get('model_path')
            
        self.saver = tf.train.Saver(max_to_keep=self.cfg.get('checkpoint_max_to_keep'))

        # Load model if referenced in the config
        if model_path is not None:
            print('Loading model from the path %s' % model_path)
            self.saver.restore(session, self.cfg.get('model_path'))

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

    def __prepare_summary_writers(self):
        # TODO
        pass

    def __load_embeddings(self):
        '''Loads the embeddings with the associated vocabulary
           and saves them for later usage in the DataLoader and
           while training/testing.'''
        if self.cfg.get('w2v_embeddings'):
            self.embeddings, self.vocabulary = utils.load_w2v_embeddings(self.cfg.get('w2v_embeddings'))
        elif self.cfg.get('ft_embeddings'):
            self.embeddings, self.vocabulary = utils.load_w2v_embeddings(self.cfg.get('ft_embeddings'))
        else:
            self.embeddings, self.vocabulary = None, {}

    def __get_model_path(self, version=0):
        '''Returns the path to store the model at as a string. An
           optional version can be specified and will be appended
           to the name of the stored file. If a model_path is set
           in the config, this will be returned and version will be
           ignored.'''
        if self.cfg.get('model_path'):
            return self.cfg.get('model_path')

        if not self.curr_exp_path:
            raise Exception('__prepare_results_directory() must be called before using __get_model_path()')

        return path.join(self.curr_exp_path, 'model-%s.chkp' % str(version))

    def __prepare_data_batch(self, all_data):
        '''Returns two lists, each of the size of the configured batch size. The first contains
           the input sentences (sentences which the first "person" said), the latter contains the
           list of expected answers.'''
        data_batch_x, data_batch_y = [], []
        batch_size = self.cfg.get('batch_size')

        while len(data_batch_y) < batch_size:
            conversation = next(all_data)

            for i, conv_turn in enumerate(conversation):
                if (i % 2) == 0:
                    data_batch_x.append(conv_turn)
                else:
                    data_batch_y.append(conv_turn)

        return data_batch_x, data_batch_y
