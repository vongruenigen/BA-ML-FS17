#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the Model class which
#              is responsible for creating the tensorflow
#              graph, i.e. building the bidirectional LSTM
#              network used in this project.
#

import utils
import logger
import math
import tensorflow as tf
import numpy as np
import model

from tensorflow.contrib import rnn, seq2seq, layers
from config import Config

class Seq2Seq(model.Base):
    '''Responsible for building the tensorflow graph, i.e.
       setting up the network so that it can be used by the
       Runner class. This implementation is heavily based on
       the tensorflow tutorial found at:
       https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/model_new.py'''

    # List of possible style of RNN cells
    CELL_FN = {
        'LSTM': rnn.LSTMCell,
        'GRU': rnn.GRUCell,
        'RNN': rnn.BasicRNNCell
    }

    # Default settings
    DEFAULT_PARAMS = {
        'cell_type': 'LSTM'
    }

    def __init__(self, cfg, session):
        '''Constructor for the seq2seq model.'''
        self.s2s_cfg = self.DEFAULT_PARAMS.copy()
        self.s2s_cfg.update(cfg.get('model_config'))
        self.cell_fn = self.CELL_FN[self.s2s_cfg.get('cell_type')]
        super(Seq2Seq, self).__init__(cfg, session)

    def get_train_op(self):
        return self._train_op

    def get_metric_ops(self):
        return {'loss': self._loss, 'perplexity': self._perplexity}

    def get_inference_op(self):
        return self.decoder_prediction_train

    def build(self):
        '''Builds sequence-to-sequence model.'''
        self.__init_cells()
        self.__init_placeholders()
        self.__init_decoder_train()
        self.__init_embeddings()

        if self.cfg.get('bidirectional'):
            self.__init_bidirectional_encoder()
        else:
            self.__init_unidirectional_encoder()

        self.__init_decoder()
        self.__init_optimizer()

    def make_train_inputs(self, input_seq, target_seq):
        '''This method is responsible for preparing the given sequences
           so that they can be used for training the model.'''
        inputs_, inputs_length_ = utils.batch(input_seq)
        targets_, targets_length_ = utils.batch(target_seq)

        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def make_inference_inputs(self, input_seq):
        '''This method is responsible for preparing the given sequences
           so that they can be used for inference using the model.'''
        inputs_, inputs_length_ = utils.batch(input_seq)

        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }

    def __init_cells(self):
        '''Initializes the encoder and decoder cells used for the model.'''
        self.encoder_cell = self.cell_fn(self.cfg.get('num_hidden_units'))
        self.decoder_cell = self.cell_fn(self.cfg.get('num_hidden_units'))

        def wrap_dropout(cell):
            return rnn.DropoutWrapper(cell, input_keep_prob=self.cfg.get('dropout_input_keep'),
                                      output_keep_prob=self.cfg.get('dropout_output_keep'))

        if self.cfg.get('dropout_input_keep') < 1.0 or self.cfg.get('dropout_output_keep') < 1.0:
            self.encoder_cell = wrap_dropout(self.encoder_cell)
            self.decoder_cell = wrap_dropout(self.decoder_cell)

        if self.cfg.get('num_encoder_layers') > 1:
            self.encoder_cell = [self.encoder_cell] * self.cfg.get('num_encoder_layers')

        if self.cfg.get('num_decoder_layers') > 1:
            self.decoder_cell = [self.decoder_cell] * self.cfg.get('num_decoder_layers')

    def __init_placeholders(self):
        '''This function is responsible for initializing the
           placeholders which are used to pass data to the enc/dec.'''
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs'
        )

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length'
        )

        # The decoder targets are only used when training, since we don't
        # need to pass any data to the decoder part when actually using it
        # to predict a sequence.
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )

        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length'
        )

    def __init_decoder_train(self):
        '''Is responsible for setting up the connections to the decoder
           which is only necessary while training, not whily using the model.'''
        with tf.name_scope('decoder_train_feeds'):
            seq_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * Config.EOS_WORD_IDX
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * Config.PAD_WORD_IDX

            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=Config.EOS_WORD_IDX,
                                                        off_value=Config.PAD_WORD_IDX,
                                                        dtype=tf.int32)

            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
            decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name='loss_weights')

    def __init_embeddings(self):
        '''Initializes the part of the model which is responsible for
           converting an input sequence to a sequence-matrix with the
           given vector embeddings.'''
        self.embeddings = self.cfg.get('embeddings')

        if self.embeddings is None or len(self.embeddings) == 0:
            logger.fatal('No embeddings set in the config, should have been '
                         'randomly initialized by the Runner?')

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_train_inputs)

    def __init_unidirectional_encoder(self):
        '''Initializes the "simple", unidirectional encoder which is responsible
           encoding the input sequence into a thought vector.'''
        with tf.name_scope('encoder_unidirectional'):
            (self.encoder_outputs, self.encoder_state) = tf.nn.dynamic_rnn(
                cell=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                time_major=True,
                dtype=tf.float32
            )

    def __init_bidirectional_encoder(self):
        '''Initializes the "complex", bidirectional encoder which is responsible
           encoding the input sequence into a thought vector.'''
        with tf.name_scope('encoder_bidirectional'):
            ((encoder_fw_outputs, encoder_bw_outputs),
             (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.encoder_cell,
                cell_bw=self.encoder_cell,
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                time_major=True,
                dtype=tf.float32
            )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, rnn.LSTMStateTuple):
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1,
                    name='bidirectional_concat_c'
                )

                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1,
                    name='bidirectional_concat_h'
                )

                self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1,
                                               name='bidirectional_concat')

    def __init_decoder(self):
        '''Initializes the decoder part of the model.'''
        with tf.variable_scope('decoder') as scope:
            output_fn = lambda outs: layers.linear(outs, self.__get_vocab_size(), scope=scope)

            if self.cfg.get('use_attention'):
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

                (attention_keys, attention_values,
                 attention_score_fn, attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option='bahdanau',
                    num_units=self.decoder_cell.output_size
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embeddings,
                    start_of_sequence_id=Config.EOS_WORD_IDX,
                    end_of_sequence_id=Config.EOS_WORD_IDX,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.__get_vocab_size()                    
                )
            else:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embeddings,
                    start_of_sequence_id=Config.EOS_WORD_IDX,
                    end_of_sequence_id=Config.EOS_WORD_IDX,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.__get_vocab_size()
                )

            (self.decoder_outputs_train, self.decoder_state_train,
                 self.decoder_context_state_train) = seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope
                )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_traion')

            scope.reuse_variables()

            (self.decoder_logits_inference, decoder_state_inference,
             self.decoder_context_state_inference) = seq2seq.dynamic_rnn_decoder(
                cell=self.decoder_cell,
                decoder_fn=decoder_fn_inference,
                time_major=True,
                scope=scope
            )

            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def __init_optimizer(self):
        '''Initializes the optimizer which should be used for the training.'''
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        softmax_loss_fn = None
        
        # If there are more words than configured in 'max_vocabulary_size',
        # we start using sampled softmax, otherwise the training process won't
        # fit on a single GPU without problems.
        if self.__get_vocab_size() > self.cfg.get('max_vocabulary_size'):
            target_vocab_size = self.cfg.get('max_vocabulary_size')
            num_hidden_units = self.cfg.get('num_hidden_units')
            num_sampled = self.cfg.get('sampled_softmax_number_of_samples')
            
            w = tf.get_variable('out_proj_w', [num_hidden_units, target_vocab_size], dtype=tf.float32)
            w_t = tf.transpose(w)
            b = tf.get_variable('out_proj_b', [target_vocab_size], dtype=tf.float32)
            
            self.output_projection = (w, b)

            def sampled_softmax_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)

                return tf.nn.sampled_softmax_loss(
                        local_w_t, local_b,
                        labels, local_inputs,
                        num_sampled, target_vocab_size)

            #softmax_loss_fn = sampled_softmax_loss

        self._loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                           weights=self.loss_weights,
                                           softmax_loss_function=softmax_loss_fn)

        self._perplexity = tf.pow(2.0, self._loss)

        self._train_op = tf.train.AdadeltaOptimizer().minimize(self._loss, global_step=self._global_step)

    def __get_vocab_size(self):
        '''Returns the size of the vocabulary.'''
        return len(self.cfg.get('vocabulary'))
