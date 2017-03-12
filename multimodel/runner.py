#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module is responsible for training
#              and running of a given model.
#

import sys
import os
import tensorflow as tf
import numpy as np
import contextlib
import utils
import json
import logger
import tqdm
import math

import seq2seq, memn2n

from os import path
from config import Config
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

        logger.init_logger(self.cfg)
        logger.debug('The following config will be used')
        logger.debug(json.dumps(self.cfg.cfg_obj, indent=4, sort_keys=True))

        self.__prepare_summary_writers()
        self.__prepare_results_directory()
        self.__load_embeddings_and_vocabulary()

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

            metrics_track = []

            # TODO: Remove testing data
            length_from=3
            length_to=8
            vocab_lower=3
            vocab_upper=10
            batch_size=100
            max_batches=5000
            batches_in_epoch=1000
            verbose=True

            batches = utils.random_sequences(length_from=length_from, length_to=length_to,
                                             vocab_lower=vocab_lower, vocab_upper=vocab_upper,
                                             batch_size=batch_size)

            training_batches = batches

            # Collect all neccesary ops to run
            metric_ops_dict = model.get_metric_ops()
            metric_names = list(metric_ops_dict.keys())
            metric_ops = [metric_ops_dict[n] for n in metric_names]
            all_ops = [model.get_train_op()] + metric_ops

            try:
                for epoch in range(self.cfg.get('epochs')):
                    epoch_nr = epoch + 1
                    feed_dict = {}

                    logger.info('[Starting epoch #%i]' % epoch_nr)
                    print()
                    
                    for batch in tqdm.tqdm(range(self.cfg.get('batches_per_epoch'))):
                        # Prepare the training data
                        batch_data_x, batch_data_y = self.__prepare_data_batch(training_batches)
                        feed_dict = model.make_train_inputs(batch_data_x, batch_data_y)
                        results = session.run(all_ops, feed_dict)

                        results.pop(0) # don't need the result from the training op

                        if len(results) > 0:
                            metrics_results = {metric_names[i]: results[i] for i, x in enumerate(metric_names)}
                            metrics_track.append(metrics_results)

                    print()
                    self.__print_epoch_state(metrics_track[-1], epoch_nr)
                    self.__show_samples_at_end_of_epoch(model, session, feed_dict, epoch_nr)

                    # Store model after each epoch
                    self.__store_model(session, model, epoch_nr)
            except KeyboardInterrupt:
                logger.warn('training interrupted')
                # TODO: Save state when the training is interrupted?

            # self.__store_metrics(loss_track, perplexity_track)

    def test(self):
        '''This method is responsible for evaluating a trained
           model with the settings defined in the config.'''
        with self.__with_model() as (session, model):
            pass

    def inference(self, text):
        '''This method is responsible for doing inference on
           a list of texts. It returns a single answer from the
           machine.'''
        with self.__with_model() as (session, model):
            text_idxs = self.data_loader.convert_text_to_indices(text, self.vocabulary)
            feed_dict = model.make_inference_inputs([text_idxs])

            answer_idxs = session.run([model.decoder_prediction_inference], feed_dict)
            answer_idxs = list(map(lambda x: x[0], answer_idxs[0]))

            return self.data_loader.convert_indices_to_text(answer_idxs, self.rev_vocabulary)

    @contextlib.contextmanager
    def __with_model(self):
        '''This method is responsible for setting up the device,
           session and model which can then be used for training
           or inference.'''
        with tf.device(self.__get_device()):
            with self.__with_tf_session() as session:
                model = self.__create_model(session)

                # Build and initialize the graph
                model.build()
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

    def __store_model(self, session, model, epoch_nr):
        '''Stores the given model in the current results directory.
           This model can then later be reloaded with the __load_model()
           method.'''
        model_path = self.__get_model_path()
        self.saver.save(session, model_path, global_step=model.get_global_step())
        logger.info('Current version of the model stored after epoch #%i' % epoch_nr)

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
        self.saver = tf.train.Saver(max_to_keep=self.cfg.get('checkpoint_max_to_keep'))
        model_path = self.cfg.get('model_path')

        # Load model if referenced in the config
        if model_path is not None:
            logger.info('Loading model from the path %s' % model_path)
            self.saver.restore(session, model_path)

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

    def __load_embeddings_and_vocabulary(self):
        '''Loads the embeddings with the associated vocabulary
           and saves them for later usage in the DataLoader and
           while training/testing.'''
        if self.cfg.get('vocabulary') is None and self.cfg.get('use_integer_vocabulary'):
            # Build a stub dictionary which maps integers to integers for the defined range
            self.vocabulary = {str(x): x for x in range(self.cfg.get('max_vocabulary_size'))}
        else:
            self.vocabulary = utils.load_vocabulary(self.cfg.get('vocabulary'))

        if self.cfg.get('w2v_embeddings'):
            self.embeddings = utils.load_w2v_embeddings(self.cfg.get('w2v_embeddings'))
        elif self.cfg.get('ft_embeddings'):
            self.embeddingsy = utils.load_ft_embeddings(self.cfg.get('ft_embeddings'))
        elif self.cfg.get('use_random_embeddings'):
            # uniform(-sqrt(3), sqrt(3)) has variance=1
            sqrt3 = math.sqrt(3)
            self.embeddings = np.random.uniform(-sqrt3, sqrt3, size=(len(self.vocabulary), 
                                                self.cfg.get('max_random_embeddings_size')))
        else:
            self.embeddings = None

        if self.embeddings is not None:
            # Prepare the vocabulary and embeddings (e.g. add embedding for unknown words)
            self.embeddings, self.vocabulary = utils.prepare_embeddings_and_vocabulary(
                                                            self.embeddings, self.vocabulary)

        # Store the embeddings and vocabulary in the 
        self.cfg.set('embeddings', self.embeddings)
        self.cfg.set('vocabulary_m', self.vocabulary)

        # revert the vocabulary for the idx -> text usages
        self.rev_vocabulary = utils.reverse_vocabulary(self.vocabulary)

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

    def __print_epoch_state(self, metrics, epoch_nr):
        '''Prints the metrics after an epoch has been finished.'''
        logger.info('[Finished Epoch #%i]' % epoch_nr)

        max_len_name = max(map(lambda x: len(x), metrics.keys()))

        for k, v in metrics.items():
            logger.info('  %s= %f' % (k.ljust(max_len_name+1), float(v)))

    def __show_samples_at_end_of_epoch(self, model, session, feed_dict, epoch_nr):
        if self.cfg.get('show_predictions_while_training'):
            logger.info('Last three samples with predictions:')

            for i, (e_in, dt_exp, dt_pred) in enumerate(zip(
                feed_dict[model.encoder_inputs].T,
                feed_dict[model.decoder_targets].T,
                session.run(model.get_inference_op(), feed_dict).T
            )):
                # dt_pred = list(map(lambda x: x - 1, dt_pred))

                if self.cfg.get('show_text_when_showing_predictions'):
                    text_in = self.data_loader.convert_indices_to_text(e_in, self.rev_vocabulary)
                    text_exp = self.data_loader.convert_indices_to_text(dt_exp, self.rev_vocabulary)
                    text_pred = self.data_loader.convert_indices_to_text(dt_pred, self.rev_vocabulary)
                else:
                    text_in = ', '.join(map(lambda x: str(x), e_in))
                    text_pred = ', '.join(map(lambda x: str(x), dt_pred))
                    text_exp = ', '.join(map(lambda x: str(x), dt_pred))

                logger.info('[Sample #%i of epoch #%i]' % (i+1, epoch_nr))
                logger.info('  Input      > %s' % utils.truncate(text_in))
                logger.info('  Prediction > %s' % utils.truncate(text_out))
                logger.info('  Expected   > %s' % utils.truncate(text_exp))
                
                # don't show more than three samples per epoch
                if i >= 2: break

    def __create_model(self, session):
        '''Creates a model based on the name given.'''
        name = self.cfg.get('model_name')
        model_ctor_map = {
            'seq2seq': seq2seq.Seq2Seq,
            'memn2n': memn2n.MemN2N
        }

        if name not in model_ctor_map:
            logger.fatal('the model with the name %s does not exist\n'
                         'the following are available: '
                         ', '.join(model_ctor_map.keys()))

        return model_ctor_map[name](self.cfg, session)

    def __prepare_data_batch(self, all_data):
        '''Returns two lists, each of the size of the configured batch size. The first contains
           the input sentences (sentences which the first "person" said), the latter contains the
           list of expected answers.'''
        data_batch_x, data_batch_y = [], []
        batch_size = self.cfg.get('batch_size')

        conv_turn_idx = 0
        conversation = next(all_data)
        first_conv_turn = None
        second_conv_turn = None

        while len(data_batch_y) < batch_size and len(data_batch_x) < batch_size:
            # Check if we've reached the end of the conversation, in this
            # case we've to load the next conversation.
            try:
                if conv_turn_idx + 1 >= len(conversation):
                    conversation = next(all_data)
                    conv_turn_idx = 0
            except StopIteration as e:
                break # exit the loop in case there is no more data  

            first_conv_turn = conversation[conv_turn_idx]
            second_conv_turn = conversation[conv_turn_idx+1]
            
            if self.cfg.get('train_on_copy'):
                data_batch_x.append(first_conv_turn)
                data_batch_y.append(first_conv_turn)
            else:
                data_batch_x.append(first_conv_turn)
                data_batch_y.append(second_conv_turn)

            conv_turn_idx += 2

        return data_batch_x, data_batch_y
