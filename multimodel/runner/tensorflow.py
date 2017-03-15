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

from os import path

from multimodel.constants import RESULTS_PATH
from multimodel.model.tensorflow import seq2seq, memn2n
from multimodel.runner.base import Base
from multimodel.config import Config
from multimodel.data_loader.conversational import ConversationalDataLoader

class TensorflowRunner(Base):
    '''This class is responsible for the training,
       testing and running of the model.'''

    def __init__(self, cfg_path):
        '''Creates a new instance of the TensorflowRunner class.'''
        super(TensorflowRunner, self).__init__(cfg_path)
        self.session = None

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

            if self.cfg.get('use_random_integer_sequences') and not training_batches:
                batches = utils.random_sequences(length_from=3, length_to=8,
                                                 vocab_lower=3, vocab_upper=10,
                                                 batch_size=self.cfg.get('batch_size'))
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

            self.store_metrics(metrics_track)

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
                if self.model is None:
                    self.__load_embeddings_and_vocabulary()

                    # Build and initialize the graph
                    self.model = self.__create_model(session)
                    self.model.build()
                    session.run(tf.global_variables_initializer())

                yield (session, self.model)

    @contextlib.contextmanager
    def __with_tf_session(self):
        '''This method is responsible for wrapping the execution
           of a given block within a tensorflow session.'''
        with tf.Graph().as_default():
            if self.session is None:
                self.session = tf.Session()

            yield self.session

    def __get_device(self):
        '''Returns the name of the device to use when executing
           a computation.'''
        return '/gpu:0'

    def __store_model(self, session, model, epoch_nr):
        '''Stores the given model in the current results directory.
           This model can then later be reloaded with the __load_model()
           method.'''
        model_path = self.get_model_path()
        self.saver.save(session, model_path, global_step=model.get_global_step())
        logger.info('Current version of the model stored after epoch #%i' % epoch_nr)

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

        embeddings_matrix = None

        if self.cfg.get('w2v_embeddings'):
            embeddings_matrix = utils.load_w2v_embeddings(self.cfg.get('w2v_embeddings'))
        elif self.cfg.get('ft_embeddings'):
            embeddings_matrix = utils.load_ft_embeddings(self.cfg.get('ft_embeddings'))
        elif self.cfg.get('use_random_embeddings'):
            max_idx = self.cfg.get('max_vocabulary_size')
            new_vocabulary = {}

            for k, v in self.vocabulary.items():
                if v <= max_idx+1:
                    new_vocabulary[k] = v

            self.vocabulary = new_vocabulary

            # uniform(-sqrt(3), sqrt(3)) has variance=1
            sqrt3 = math.sqrt(3)
            embeddings_matrix = np.random.uniform(-sqrt3, sqrt3, size=(len(self.vocabulary), 
                                                  self.cfg.get('max_random_embeddings_size')))

        if embeddings_matrix is not None:
            # Prepare the vocabulary and embeddings (e.g. add embedding for unknown words)
            embeddingx_matrix, self.vocabulary = utils.prepare_embeddings_and_vocabulary(
                                                            embeddings_matrix, self.vocabulary)

        self.embeddings = tf.get_variable(name='embeddings_m', shape=embeddings_matrix.shape,
                                          trainable=self.cfg.get('train_word_embeddings'),
                                          initializer=tf.constant_initializer(embeddings_matrix))

        # Store the embeddings and vocabulary in the 
        self.cfg.set('embeddings', self.embeddings)
        self.cfg.set('vocabulary_m', self.vocabulary)

        # revert the vocabulary for the idx -> text usages
        self.rev_vocabulary = utils.reverse_vocabulary(self.vocabulary)

    def __print_epoch_state(self, metrics, epoch_nr):
        '''Prints the metrics after an epoch has been finished.'''
        logger.info('[Finished Epoch #%i]' % epoch_nr)

        max_len_name = max(map(lambda x: len(x), metrics.keys()))

        for k, v in metrics.items():
            logger.info('  %s= %f' % (k.ljust(max_len_name+1), float(v)))

    def __show_samples_at_end_of_epoch(self, model, session, feed_dict, epoch_nr):
        if self.cfg.get('show_predictions_while_training'):
            logger.info('Last three samples with predictions:')

            text_in = None
            text_exp = None
            text_pred = None

            for i, (e_in, dt_exp, dt_pred) in enumerate(zip(
                feed_dict[model.encoder_inputs].T,
                feed_dict[model.decoder_targets].T,
                session.run(model.get_inference_op(), feed_dict).T
            )):
                if self.cfg.get('show_text_when_showing_predictions'):
                    text_in = self.data_loader.convert_indices_to_text(e_in, self.rev_vocabulary)
                    text_exp = self.data_loader.convert_indices_to_text(dt_exp, self.rev_vocabulary)
                    text_pred = self.data_loader.convert_indices_to_text(dt_pred, self.rev_vocabulary)
                else:
                    text_in = ', '.join(map(lambda x: str(int(x)), e_in))
                    text_pred = ', '.join(map(lambda x: str(int(x)), dt_pred))
                    text_exp = ', '.join(map(lambda x: str(int(x)), dt_exp))

                logger.info('[Sample #%i of epoch #%i]' % (i+1, epoch_nr))
                logger.info('  Input      > %s' % utils.truncate(text_in, width=100))
                logger.info('  Prediction > %s' % utils.truncate(text_pred, width=100))
                logger.info('  Expected   > %s' % utils.truncate(text_exp, width=100))
                
                # don't show more than three samples per epoch
                if i > 2: break

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
