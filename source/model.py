#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains the Model class which
#              is responsible for creating the tensorflow
#              graph, i.e. building the bidirectional LSTM
#              network used in this project.
#

import utils
import math
import tensorflow as tf

from tensorflow.contrib import rnn, seq2seq, layers

class Model(object):
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

    # Tokens for the PAD and EOS symbols
    PAD_TOKEN = 0
    EOS_TOKEN = 1
    
    def __init__(self, cfg):
        '''Constructor for the Model class. It expects only a
           config object to be given as the first parameter.'''
        self.cfg = cfg
        self.cell_fn = self.CELL_FN[self.cfg.get('cell_type')]
    
    def build(self):
        '''Must be called after setting up the model object
           (e.g. embeddings provided, etc) BEFORE any usage
           in a training or inference scenario.'''
        self.__build_model()

    def get_global_step(self):
        '''Returns the current global step counter.'''
        return self.global_step

    def get_embeddings(self):
        '''Returns the embeddings matrix.'''
        return self.embeddings

    def set_embeddings(self, emb):
        '''Sets the embeddings matrix for the model.'''
        self.embeddings = emb

    def train(self, session, train_inputs, train_outputs):
        '''This method is responsible for training the model.'''
        pass

    def inference(self, session, inputs):
        ''''''
        pass

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

    def __build_model(self):
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

    def __init_cells(self):
        '''Initializes the encoder and decoder cells used for the model.'''
        self.encoder_cell = self.cell_fn(self.cfg.get('num_hidden_units'))
        self.decoder_cell = self.cell_fn(self.cfg.get('num_hidden_units'))

        def wrap_droput(cell):
            return rnn.DropoutWrapper(cell, input_keep_prob=self.cfg.get('dropout_input_keep'),
                                      output_keep_prob=self.cfg.get('dropout_output_keep'))

        if self.cfg.get('dropout_input_keep') < 1.0 or self.cfg.get('dropout_output_keep') < 1.0:
            self.encoder_cell = wrap_droput(self.encoder_cell)
            self.decoder_cell = wrap_droput(self.decoder_cell)

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

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS_TOKEN
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD_TOKEN

            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS_TOKEN,
                                                        off_value=self.PAD_TOKEN,
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
        if self.embeddings is not None and len(self.embeddings) > 0:
            print('No embeddings given, initializing random embeddings')

            initializer = tf.random_uniform_initializer(-1, 1)

            self.embeddings = tf.get_variable(
                name='embeddings',
                shape=[self.cfg.get('max_vocabulary_size'), 10],
                initializer=initializer,
                dtype=tf.float32
            )

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
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

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
                    start_of_sequence_id=self.EOS_TOKEN,
                    end_of_sequence_id=self.EOS_TOKEN,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.__get_vocab_size()                    
                )
            else:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embeddings,
                    start_of_sequence_id=self.EOS_TOKEN,
                    end_of_sequence_id=self.EOS_TOKEN,
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

        # Track the global step state when training
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

    def __get_vocab_size(self):
        '''Returns the size of the vocabulary if the embeddings are
           already loaded, otherwise an error is thrown.'''
        if not self.get_embeddings():
            raise Exception('embeddings must be set via set_embeddings()')
        else:
            return self.embeddings.get_shape()[0].value
