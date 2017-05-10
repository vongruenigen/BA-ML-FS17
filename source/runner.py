#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
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

from model import TSeq2SeqModel
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

        self.__graph = None
        self.__session = None
        self.__model = None

    def close(self):
        '''Destroys all ressources acquired by this runner.'''
        if self.__session:
            self.__session.close()

            self.__session = None
            self.__graph = None
            self.__model = None

    def train(self):
        '''This method is responsible for training a model
           with the settings defined in the config.'''
        with self.__with_model() as (session, model):
            self.__store_config()

            training_batches = []
            validation_batches = []

            self.__update_global_step(session, model)

            if self.config.get('training_data'):
                training_batches = self.data_loader.load_conversations(
                    self.config.get('training_data'),
                    self.config.get('vocabulary_dict')
                )

            loss_track = []
            val_loss_track = []
            perplexity_track = []
            val_perplexity_track = []
            curr_min_perplexity = float('inf')

            if self.config.get('metrics') is not None:
                logger.info('Loading existing metrics before starting the training')
                metrics = self.config.get('metrics')
                loss_track = metrics['loss']
                val_loss_track = metrics['val_loss']
                perplexity_track = metrics['perplexity']
                val_perplexity_track = metrics['val_perplexity']
                curr_min_perplexity = min(val_perplexity_track)

            epochs_per_validation = self.config.get('epochs_per_validation')

            try:
                for epoch_nr in range(1, self.config.get('epochs')+1):
                    self.__update_global_step(session, model)

                    # Run one epoch and get the resulting loss and perplexity
                    loss, perplexity, last_batch = self.__run_epoch(session, model, training_batches, epoch_nr)

                    loss_track.append(loss)
                    perplexity_track.append(perplexity)

                    # Show predictions & store the model after one epoch
                    self.__show_predictions(session, model, last_batch, epoch_nr)

                    if epoch_nr % epochs_per_validation == 0 and self.config.get('validation_data'):
                        validation_batches = self.data_loader.load_conversations(
                            self.config.get('validation_data'),
                            self.config.get('vocabulary_dict'),
                            disable_forwarding=True
                        )

                        val_loss, val_perplexity, _ = self.__run_eval(session, model, validation_batches, epoch_nr)

                        val_loss_track.append(val_loss)
                        val_perplexity_track.append(val_perplexity)

                        # Store metrics each time we evaluate the model
                        self.__store_metrics(loss_track, perplexity_track, val_loss_track, val_perplexity_track)
                    else:
                        logger.info('Skipping validation since %i mod %i != 0' % (epoch_nr, epochs_per_validation))

                    if len(val_perplexity_track) > 0 and val_perplexity_track[-1] < curr_min_perplexity:
                        logger.info('Storing model since the validation perplexity improved from %f to %f' % (
                            curr_min_perplexity, val_perplexity_track[-1]
                        ))

                        self.__store_model(session, model, epoch_nr, tag='validation')
                        curr_min_perplexity = val_perplexity_track[-1]

                    self.__store_model(session, model, epoch_nr)
            except KeyboardInterrupt:
                logger.warn('Training interrupted by user')

            # Store the final metrics
            self.__store_metrics(loss_track, perplexity_track, val_loss_track, val_perplexity_track)

    def inference(self, text, trim_eos_pad=True, additional_tensor_names=[]):
        '''This method is responsible for doing inference on
           a list of texts. It returns a single answer from the
           machine.'''
        with self.__with_model() as (session, model):
            use_beam_search = self.config.get('use_beam_search')
            beam_size = self.config.get('beam_size')
            vocabulary = self.config.get('vocabulary_dict')
            text_idxs = self.data_loader.convert_text_to_indices(text, vocabulary)
            feed_dict, bucket_id = model.make_inference_inputs([text_idxs])
            inference_op = model.get_inference_op(bucket_id)
            additional_tensors = []

            if additional_tensor_names is not None and len(additional_tensor_names) > 0:
                default_graph = tf.get_default_graph()

                for name in additional_tensor_names:
                    curr_tensor = default_graph.get_tensor_by_name(name)
                    additional_tensors.append(curr_tensor)

            output_list = session.run(inference_op + additional_tensors, feed_dict)

            additional_tensors_results = None

            if len(additional_tensors) > 0:
                additional_tensors_results = output_list[-len(additional_tensors):]

            output_list = output_list[0:len(output_list)-len(additional_tensors)]

            if use_beam_search:
                beam_path = output_list[0][0]
                beam_symbol = output_list[1]
                log_beam_probs = output_list[2]
                output_logits = output_list[2:]

                beam_paths = [[] for _ in range(beam_size)]
                curr = list(range(beam_size))
                num_steps = len(beam_path)

                for i in range(num_steps-1, -1, -1):
                    for j in range(beam_size):
                        beam_paths[j].append(beam_symbol[i][curr[j]])
                        curr[j] = beam_path[i][curr[j]]

                replies = []

                for i in range(beam_size):
                    answer_idxs = [int(l) for l in beam_paths[i][::-1]]
                    reply = self.data_loader.convert_indices_to_text(answer_idxs, self.rev_vocabulary,
                                                                     trim_eos_pad=trim_eos_pad)
                    replies.append(reply)

                answer = 'Replies:\n%s' % ''.join(map(lambda x: '- %s\n' % x, replies))
            else:
                answer_idxs = np.array(output_list).transpose([1, 0, 2])
                answer_idxs = np.argmax(answer_idxs, axis=2)[0]
                answer = self.data_loader.convert_indices_to_text(answer_idxs, self.rev_vocabulary,
                                                                  trim_eos_pad=trim_on_eos)

            return (answer, additional_tensors_results)

    def __update_global_step(self, session, model):
        '''Updates the global_step value in the config.'''
        curr_global_step = session.run(model.global_step)
        self.config.set('global_step', curr_global_step)

    def __run_eval(self, session, model, validation_batches, epoch_nr):
        '''This method is responsible for evaluating a trained
           model with the settings defined in the config. It returns
           the loss on the test batch and the respecting perplexity'''
        feed_dict = {}

        batches_per_validation = self.config.get('batches_per_validation')

        if batches_per_validation <= 0:
            logger.info('Skipping evaluation because batches_per_validation is set to zero')
            return

        val_batch_data_x = None
        val_batch_data_y = None

        val_sum_losses = 0.0
        val_sum_iters = 0

        logger.info('[Starting validation for epoch #%i]' % epoch_nr)

        for batch in tqdm.tqdm(range(batches_per_validation)):
            val_batch_data_x, val_batch_data_y = self.__prepare_data_batch(validation_batches)
            feed_dict, bucket_id = model.make_train_inputs(val_batch_data_x, val_batch_data_y)

            loss_op = model.get_loss_op(bucket_id)
            val_loss = session.run(loss_op, feed_dict)

            val_sum_losses += val_loss
            val_sum_iters += 1

        val_avg_loss = val_sum_losses / val_sum_iters
        val_perplexity = self.__calculate_perplexity(val_avg_loss)

        logger.info('[Finished validation after #%i]' % epoch_nr)
        logger.info('    Validation Loss       > %f' % val_loss)
        logger.info('    Validation Perplexity > %f' % val_perplexity)

        return val_avg_loss, val_perplexity, (val_batch_data_x, val_batch_data_y)

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
        perplexity = self.__calculate_perplexity(avg_loss)

        logger.info('[Finished epoch #%i]' % epoch_nr)
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
                if self.config.get('reverse_input'): e_in = reversed(e_in)

                text_in = self.data_loader.convert_indices_to_text(e_in, self.rev_vocabulary)
                text_exp = self.data_loader.convert_indices_to_text(dt_exp, self.rev_vocabulary)
                text_out = self.data_loader.convert_indices_to_text(dt_pred, self.rev_vocabulary)

                logger.info('[sample #%i of epoch #%i]' % (i+1, epoch_nr))
                logger.info('    input      > %s' % text_in)
                logger.info('    prediction > %s' % text_out)
                logger.info('    expected   > %s' % text_exp)

    @contextlib.contextmanager
    def __with_model(self):
        '''This method is responsible for setting up the device,
           session and model which can then be used for training
           or inference.'''
        with tf.device(self.config.get('device')):
            with self.__with_tf_session() as session:
                if self.__model is None:
                    self.__model = TSeq2SeqModel(self.config)
                    self.__model.build()
                    self.__setup_saver_and_restore_model(session)

                yield (session, self.__model)

    def __calculate_perplexity(self, loss):
        '''Returns the perplexity for the given loss value.'''
        return np.power(2, loss)

    @contextlib.contextmanager
    def __with_tf_session(self):
        '''This method is responsible for wrapping the execution
           of a given block within a tensorflow session.'''
        if self.__graph is None or self.__session is None:
            self.__graph = tf.Graph().as_default()
            self.__session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with self.__session.as_default():
            yield self.__session

    @contextlib.contextmanager
    def __with_tf_saver(self, session):
        '''This method is responsible for ensuring that the state
           of the model is saved at all times using the tf.Saver
           class'''
        pass

    def __store_config(self):
        '''Stores the config in the results directory.'''
        config_path = path.join(self.curr_exp_path, 'config.json')

        with open(config_path, 'w+') as f:
            json.dump(self.config.get_dict(), f,
                      indent=4, sort_keys=True)

    def __store_model(self, session, model, epoch_nr, tag=''):
        '''Stores the given model in the current results directory.
           This model can then later be reloaded with the __load_model()
           method.'''
        if (epoch_nr % self.config.get('save_model_after_n_epochs') == 0) or tag != '':
            model_path = self.__get_model_path(tag=tag)
            self.saver.save(session, model_path, global_step=model.global_step)
            logger.info('Current version of the model stored after epoch #%i' % epoch_nr)

    def __store_metrics(self, loss_track, perplexity_track, val_loss_track, val_perplexity_track):
        '''This method is responsible for storing the metrics
           collected while training the model. Currently, this
           only includes the losses and perplexities.'''
        metrics_path = path.join(self.curr_exp_path, 'metrics.json')

        loss_track = list(map(float, loss_track))
        perplexity_track = list(map(float, perplexity_track))

        val_loss_track = list(map(float, val_loss_track))
        val_perplexity_track = list(map(float, val_perplexity_track))

        with open(metrics_path, 'w+') as f:
            json.dump({'loss': loss_track,
                       'val_loss': val_loss_track,
                       'perplexity': perplexity_track,
                       'val_perplexity': val_perplexity_track},
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
            model_dir = None

            if path.isdir(model_path):
                ckpt = tf.train.get_checkpoint_state(self.config.get('model_path'))

                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(session, ckpt.model_checkpoint_path)

                model_dir = model_path
            else:
                self.saver.restore(session, model_path)
                model_dir = '/'.join(model_path.split('/')[:-1])

            metrics_path = path.join(model_dir, 'metrics.json')

            if path.isfile(metrics_path):
                logger.info('Loading metrics from %s' % metrics_path)

                with open(metrics_path, 'r') as f:
                    self.config.set('metrics', json.load(f))

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

    def __get_model_path(self, version=0, tag=''):
        '''Returns the path to store the model at as a string. An
           optional version can be specified and will be appended
           to the name of the stored file. If a model_path is set
           in the config, this will be returned and version will be
           ignored.'''
        if not self.curr_exp_path:
            raise Exception('__prepare_results_directory() must be called before using __get_model_path()')

        model_name = None

        if tag == '':
            model_name = 'model-%s.chkp' % str(version)
        else:
            model_name = 'model-%s-%s.chkp' % (str(tag), str(version))

        return path.join(self.curr_exp_path, model_name)

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
