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
import logger
import tqdm

from model import Model, PSeq2SeqModel, TSeq2SeqModel
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
            self.config_path = cfg_path
            self.config = Config.load_from_json(self.config_path)
        elif isinstance(cfg_path, Config):
            self.config_path = None
            self.config = cfg_path
        else:
            raise Exception('cfg_path must be either a path or a Config object')

        logger.init_logger(self.config)

        self.__prepare_summary_writers()
        self.__prepare_results_directory()
        self.__load_embeddings()

        self.data_loader = DataLoader(self.config)

    def train(self):
        '''This method is responsible for training a model
           with the settings defined in the config.'''
        with self.__with_model() as (session, model):
            #self.__setup_saver_and_restore_model(session)

            training_batches = []
            test_batches = []

            if self.config.get('training_data'):
                training_batches = self.data_loader.load_conversations(
                    self.config.get('training_data'),
                    self.config.get('vocabulary_dict')
                )

            if self.config.get('test_data'):
                test_batches = self.data_loader.load_conversations(
                    self.config.get('test_data'),
                    self.config.get('vocabulary_dict')
                )
            
            loss_track = []
            perplexity_track = []

            try:
                for epoch_nr in range(1, self.config.get('epochs')+1):                    
                    # Run one epoch and get the resulting loss and perplexity
                    loss, perplexity, last_batch = self.__run_epoch(session, model, training_batches, epoch_nr)

                    loss_track.append(loss)
                    perplexity_track.append(perplexity)

                    # Decay learning rate in case that there was no progress for the last three epochs
                    if len(loss_track) % 3 == 0 and len(loss_track) > 0:
                        best_loss = min(loss_track)

                        if all(map(lambda x: x > best_loss, loss_track[-3:])):
                            logger.info('Decaying learning rate because there was no progress in the last three epochs')
                            session.run(model.learning_rate_decay_op)
                            logger.info('The new learning rate is %.5f' % model.learning_rate.eval())
                    
                    # Show predictions & store the model after one epoch
                    self.__show_predictions(session, model, last_batch, epoch_nr)
                    self.__store_model(session, model, epoch_nr)
            except KeyboardInterrupt:
                logger.warn('training interrupted')
                # TODO: Save state when the training is interrupted?

            self.__store_metrics(loss_track, perplexity_track)

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
            vocabulary = self.config.get('vocabulary_dict')
            text_idxs = self.data_loader.convert_text_to_indices(text, vocabulary)
            feed_dict, bucket_id = model.make_inference_inputs([text_idxs])
            inference_op = model.get_inference_op(bucket_id)

            answer_idxs = session.run(inference_op, feed_dict)
            answer_idxs = np.array(answer_idxs).transpose([1, 0, 2])
            answer_idxs = np.argmax(answer_idxs, axis=2)[0]

            return self.data_loader.convert_indices_to_text(answer_idxs, self.rev_vocabulary)

    def __run_epoch(self, session, model, training_batches, epoch_nr):
        '''Runs one epoch of the training.'''
        feed_dict = {}

        batch_data_x = None
        batch_data_y = None
        
        sum_losses = 0.0
        sum_iters = 0

        logger.info('[Starting epoch #%i]' % epoch_nr)
        
        for batch in tqdm.tqdm(range(self.config.get('batches_per_epoch'))):
            batch_data_x, batch_data_y = self.__prepare_data_batch(training_batches)

            feed_dict, bucket_id = model.make_train_inputs(batch_data_x, batch_data_y)

            all_ops = model.get_train_ops(bucket_id)
            all_ops.append(model.get_loss_op(bucket_id))

            op_results = session.run(all_ops, feed_dict)

            loss = op_results[-1]

            sum_losses += loss
            sum_iters += 1

        avg_loss = sum_losses / sum_iters
        perplexity = np.power(avg_loss, 2)

        logger.info('[Finished Epoch #%i]' % epoch_nr)
        logger.info('    Loss       > %f' % loss)
        logger.info('    Perplexity > %f' % perplexity)

        return loss, perplexity, (batch_data_x, batch_data_y)

    def __show_predictions(self, session, model, last_batch, epoch_nr):
        if self.config.get('show_predictions_while_training'):
            num_show_samples    = self.config.get('batch_size')
            max_input_length    = self.config.get('max_input_length')
            max_output_length   = self.config.get('max_output_length')
            input_samples_idxs  = []
            output_samples_idxs = []

            batch_data_x, batch_data_y = last_batch

            for r in range(num_show_samples):
                sample_idxs = []

                for c in range(max_input_length):
                    curr_idx = batch_data_x[r][c]
                    sample_idxs.append(curr_idx)

                input_samples_idxs.append(sample_idxs)

            for r in range(num_show_samples):
                sample_idxs = []

                for c in range(max_output_length):
                    curr_idx = batch_data_y[r][c]
                    sample_idxs.append(curr_idx)

                output_samples_idxs.append(sample_idxs)

            num_show_samples = self.config.get('show_predictions_while_training_num')
            prediction_batch = last_batch[0]

            if num_show_samples > 0 and num_show_samples < self.config.get('batch_size'):
                prediction_batch = prediction_batch[:num_show_samples]

            feed_dict, bucket_id = model.make_inference_inputs(prediction_batch)
            inference_op = model.get_inference_op(bucket_id)

            predicted_idxs = session.run(inference_op, feed_dict)
            predicted_idxs = np.array(predicted_idxs).transpose([1, 0, 2])
            predicted_idxs = np.argmax(predicted_idxs, axis=2)

            inp_out_pred = zip(input_samples_idxs, output_samples_idxs, predicted_idxs)

            for i, (e_in, dt_exp, dt_pred) in enumerate(inp_out_pred):
                text_in = self.data_loader.convert_indices_to_text(e_in, self.rev_vocabulary)
                text_exp = self.data_loader.convert_indices_to_text(dt_exp, self.rev_vocabulary)
                text_out = self.data_loader.convert_indices_to_text(dt_pred, self.rev_vocabulary)

                logger.info('Last three samples with predictions:')
                logger.info('[Sample #%i of epoch #%i]' % (i+1, epoch_nr))
                logger.info('    Input      > %s' % text_in)
                logger.info('    Prediction > %s' % text_out)
                logger.info('    Expected   > %s' % text_exp)

    @contextlib.contextmanager
    def __with_model(self):
        '''This method is responsible for setting up the device,
           session and model which can then be used for training
           or inference.'''
        with tf.device(self.config.get('device')):
            with self.__with_tf_session() as session:
                model = TSeq2SeqModel(self.config)
                model.build()

                self.__setup_saver_and_restore_model(session)

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

    def __store_model(self, session, model, epoch_nr):
        '''Stores the given model in the current results directory.
           This model can then later be reloaded with the __load_model()
           method.'''
        if epoch_nr % self.config.get('save_model_after_n_epochs') == 0:
            model_path = self.__get_model_path()
            self.saver.save(session, model_path, global_step=epoch_nr)
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
        model_path = self.config.get('model_path')
            
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.get('checkpoint_max_to_keep'))

        # Load model if referenced in the config, otherwise freshly initialize it
        if model_path is None:
            session.run(tf.global_variables_initializer())
        else:
            logger.info('Loading model from the path %s' % model_path)
            ckpt = tf.train.get_checkpoint_state(self.config.get('model_path'))

            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(session, ckpt.model_checkpoint_path)

    def __prepare_results_directory(self):
        '''This method is responsible for preparing the results
           directory for the experiment with loaded config.'''
        if not self.config.get('train'):
            return

        if not path.isdir(Config.RESULTS_PATH):
            os.mkdir(Config.RESULTS_PATH)

        self.curr_exp_path = path.join(Config.RESULTS_PATH, self.config.get('id'))

        if path.isdir(self.curr_exp_path):
            raise Exception('A results directory with the name' +
                            ' %s already exists' % str(self.config.id))
        else:
            os.mkdir(self.curr_exp_path)

    def __prepare_summary_writers(self):
        # TODO
        pass

    def __load_embeddings(self):
        '''Loads the embeddings with the associated vocabulary
           and saves them for later usage in the DataLoader and
           while training/testing.'''
        vocabulary = None
        embeddings = None

        vocabulary = utils.load_vocabulary(self.config.get('vocabulary'))

        if self.config.get('w2v_embeddings'):
            embeddings = utils.load_w2v_embeddings(self.config.get('w2v_embeddings'))
        elif self.config.get('ft_embeddings'):
            embeddingsy = utils.load_ft_embeddings(self.config.get('ft_embeddings'))
        else:
            embeddings = np.random.uniform(
                -1.0, 1.0,
                size=(len(vocabulary),
                      self.config.get('max_random_embeddings_size'))
            )

        # Prepare the vocabulary and embeddings (e.g. add embedding for unknown words)
        embeddings, vocabulary = utils.prepare_embeddings_and_vocabulary(embeddings, vocabulary)

        self.config.set('vocabulary_dict', vocabulary)
        self.config.set('embeddings_matrix', embeddings)

        # revert the vocabulary for the idx -> text usages
        self.rev_vocabulary = utils.reverse_vocabulary(vocabulary)

    def __get_model_path(self, version=0):
        '''Returns the path to store the model at as a string. An
           optional version can be specified and will be appended
           to the name of the stored file. If a model_path is set
           in the config, this will be returned and version will be
           ignored.'''
        if not self.curr_exp_path:
            raise Exception('__prepare_results_directory() must be called before using __get_model_path()')

        return path.join(self.curr_exp_path, 'model-%s.chkp' % str(version))

    def __prepare_data_batch(self, all_data):
        '''Returns two lists, each of the size of the configured batch size. The first contains
           the input sentences (sentences which the first "person" said), the latter contains the
           list of expected answers.'''
        data_batch_x, data_batch_y = [], []
        batch_size = self.config.get('batch_size')

        conversation = next(all_data)
        conv_turn_idx = 0

        while len(data_batch_y) < batch_size and len(data_batch_x) < batch_size:
            first_conv_turn = conversation[conv_turn_idx]
            second_conv_turn = conversation[conv_turn_idx+1]

            if self.config.get('train_on_copy'):
                data_batch_x.append(first_conv_turn)
                data_batch_y.append(first_conv_turn)
            else:
                data_batch_x.append(first_conv_turn)
                data_batch_y.append(second_conv_turn)

            conv_turn_idx += 2

            # Check if we've reached the end of the conversation, in this
            # case we've to load the next conversation. In case there is
            # no conversation left, we simply exit the loop and return the
            # already loaded data.
            try:
                if conv_turn_idx == len(conversation):
                    conversation = next(all_data)
                    conv_turn_idx = 0
            except StopIteration as e:
                break # exit the loop

        return data_batch_x, data_batch_y
