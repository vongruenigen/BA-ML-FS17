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

from tensorflow.contrib import rnn, seq2seq, layers
from config import Config

class TSeq2SeqModel(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        self.__build_model()

    def __build_model(self):
        max_inp_len = self.cfg.get('max_input_length')
        max_out_len = self.cfg.get('max_output_length')
        hidden_units = 32 #self.cfg.get('num_hidden_units')
        vocab_len = len(self.cfg.get('vocabulary_dict'))
        embeddings_m = self.cfg.get('embeddings_matrix')
        embeddings_size = self.cfg.get('max_random_embeddings_size')
        buckets = self.cfg.get('buckets')
        dtype = tf.float32

        # TODO: Make configurable!
        num_samples = 512
        num_layers = 3
        learning_rate = 0.0001
        learning_rate_decay_factor = 0.99
        use_lstm = True
        max_gradient_norm = 5.0

        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < vocab_len:
          w_t = tf.get_variable('proj_w', [vocab_len, hidden_units], dtype=dtype)
          w = tf.transpose(w_t)
          b = tf.get_variable('proj_b', [vocab_len], dtype=dtype)
          output_projection = (w, b)

          def sampled_loss(labels, logits):
            labels = tf.reshape(labels, [-1, 1])
            # We need to compute the sampled_softmax_loss using 32bit floats to
            # avoid numerical instabilities.
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(logits, tf.float32)
            return tf.cast(
                tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=num_samples,
                    num_classes=vocab_len),
                dtype)
          softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
          return tf.contrib.rnn.GRUCell(hidden_units)

        def wrap_dropout(cell):
            return rnn.DropoutWrapper(cell, input_keep_prob=self.cfg.get('dropout_input_keep'),
                                            output_keep_prob=self.cfg.get('dropout_output_keep'))
        
        if use_lstm:
          def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_units)
        
        cell = single_cell()

        if self.cfg.get('train'):
            cell = wrap_dropout(cell)
        
        if num_layers > 1:
          cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols=vocab_len,
              num_decoder_symbols=vocab_len,
              embedding_size=hidden_units,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if not self.cfg.get('train'):
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=softmax_loss_function)
          
          # If we use output projection, we need to project outputs for decoding.
          if output_projection is not None:
            for b in range(len(buckets)):
              self.outputs[b] = [
                  tf.matmul(output, output_projection[0]) + output_projection[1]
                  for output in self.outputs[b]
              ]
        else:
          self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()

        self.gradient_norms = []
        self.updates = []

        if self.cfg.get('train'):
            opt = tf.train.AdamOptimizer(self.learning_rate)
            
            for b in range(len(buckets)):
              gradients = tf.gradients(self.losses[b], params)
              clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                               max_gradient_norm)
              self.gradient_norms.append(norm)
              self.updates.append(opt.apply_gradients(
                  zip(clipped_gradients, params), global_step=self.global_step))

        self.train_op = self.updates

    def make_train_inputs(self, input_seq, target_seq):
        batch_size = self.cfg.get('batch_size')
        max_out_len = self.cfg.get('max_output_length')
        feed_dict = {}

        buckets = self.cfg.get('buckets')
        bucket_id = self.__get_bucket_id(input_seq)

        if bucket_id == -1:
            raise Exception('No suitable bucket found for samples of length %i in %s' % (
                            len(input_seq[0]), buckets))

        batch_weights = []
        encoder_size, decoder_size = buckets[bucket_id]

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for i in range(len(target_seq)):
            current_input_seq = input_seq[i]
            current_target_seq = target_seq[i]

            if len(current_input_seq) < self.cfg.get('max_input_length'):
                max_input_length = self.cfg.get('max_input_length')
                padding_parts = [Config.PAD_WORD_IDX for i in range(max_input_length - len(current_input_seq))]
                current_input_seq += padding_parts

            if self.cfg.get('reverse_input'):
              current_input_seq = list(reversed(current_input_seq))

            current_target_seq.append(Config.EOS_WORD_IDX)

            if len(current_target_seq) < self.cfg.get('max_output_length'):
                max_output_length = self.cfg.get('max_output_length')
                padding_parts = [Config.PAD_WORD_IDX for i in range(max_input_length - len(current_input_seq))]
                current_target_seq += padding_parts
            
            current_target_seq.insert(0, Config.GO_WORD_IDX)

            if len(current_target_seq) > max_out_len:
                current_target_seq = current_target_seq[:max_out_len]
                current_target_seq[-1] = Config.EOS_WORD_IDX

            input_seq[i] = current_input_seq
            target_seq[i] = current_target_seq

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([input_seq[batch_idx][length_idx]
                          for batch_idx in range(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([target_seq[batch_idx][length_idx]
                          for batch_idx in range(batch_size)], dtype=np.int32))
  
            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
  
            for batch_idx in range(batch_size):
              # We set weight to 0 if the corresponding target is a PAD symbol.
              # The corresponding target is decoder_input shifted by 1 forward.
              if length_idx < decoder_size - 1:
                target = target_seq[batch_idx][length_idx + 1]
              if length_idx == decoder_size - 1 or target == Config.PAD_WORD_IDX:
                batch_weight[batch_idx] = 0.0

            batch_weights.append(batch_weight)

        for i in range(encoder_size):
            feed_dict[self.encoder_inputs[i].name] = batch_encoder_inputs[i]

        for i in range(decoder_size):
            feed_dict[self.decoder_inputs[i].name] = batch_decoder_inputs[i]
            feed_dict[self.target_weights[i].name] = batch_weights[i]

        last_target = self.decoder_inputs[decoder_size].name
        feed_dict[last_target] = np.zeros([batch_size], dtype=np.int32)

        return feed_dict, bucket_id

    def get_loss_op(self, bucket_id):
        '''Returns the loss op for the given bucket id from the current model.'''
        return self.losses[bucket_id]

    def get_inference_op(self, bucket_id):
        '''Returns the op which can be used for inference using the current model.'''
        _, decoder_size = self.cfg.get('buckets')[bucket_id]
        outputs_list = []

        for i in range(decoder_size):
            outputs_list.append(self.outputs[bucket_id][i])

        return outputs_list

    def get_train_ops(self, bucket_id):
        '''Returns the train op for the given bucket id from the current model.'''
        return [self.updates[bucket_id], self.gradient_norms[bucket_id]]

    def make_inference_inputs(self, input_seq):
        '''This method is responsible for preparing the given sequences
           so that they can be used for inference using the model.'''
        fake_outputs = []

        for i in range(len(input_seq)):
            fake_outputs.append([Config.GO_WORD_IDX] + [Config.PAD_WORD_IDX] * (len(input_seq[0]) - 1))

        return self.make_train_inputs(input_seq, fake_outputs)

    def __get_bucket_id(self, input_seq):
        '''Returns the correct bucket id to use for the length of the given examples.'''
        bucket_id = -1
        input_length = len(input_seq[0])

        for i, bucket in enumerate(self.cfg.get('buckets')):
            if bucket[0] >= input_length:
                bucket_id = i
                break

        return bucket_id

class PSeq2SeqModel(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        self.__build_model()

    def __build_model(self):
        max_inp_len = self.cfg.get('max_input_length')
        max_out_len = self.cfg.get('max_output_length')
        hidden_units = self.cfg.get('num_hidden_units')
        vocab_len = len(self.cfg.get('vocabulary_dict'))
        embeddings_m = self.cfg.get('embeddings_matrix')
        embeddings_size = self.cfg.get('max_random_embeddings_size')

        self.encoder_inputs = [tf.placeholder(shape=[None,], 
                               dtype=tf.int64, 
                               name='enc_inp_{}'.format(t)) for t in range(max_inp_len)]

        #  labels that represent the real outputs
        self.labels = [tf.placeholder(shape=[None,], 
                       dtype=tf.int64, 
                       name='target_{}'.format(t)) for t in range(max_out_len)]

        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [ tf.zeros_like(self.encoder_inputs[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)

        # define the basic cell
        basic_cell = tf.nn.rnn_cell.DropoutWrapper(
            tf.nn.rnn_cell.BasicLSTMCell(hidden_units, state_is_tuple=True),
            output_keep_prob=self.keep_prob
        )

        # stack cells together : n layered model
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*3, state_is_tuple=True)

        # for parameter sharing between training model
        #  and testing model
        with tf.variable_scope('decoder') as scope:
            # build the seq2seq model 
            #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
            self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(
                self.encoder_inputs,
                self.decoder_inputs,
                stacked_lstm,
                vocab_len, vocab_len, embeddings_size)

            # share parameters
            scope.reuse_variables()

            # testing model, where output of previous timestep is fed as input 
            #  to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                self.encoder_inputs,
                self.decoder_inputs,
                stacked_lstm,
                vocab_len,
                vocab_len,
                embeddings_size,
                feed_previous=True
            )

        loss_weights = [tf.ones_like(label, dtype=tf.float32) for label in self.labels]
        self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, vocab_len)

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def make_train_inputs(self, input_seq, target_seq):
        max_input_length = self.cfg.get('max_input_length')
        max_output_length = self.cfg.get('max_output_length')

        input_seq_batched, _ = utils.batch(input_seq, max_sequence_length=max_input_length)
        target_seq_batched, _ = utils.batch(target_seq, max_sequence_length=max_output_length)

        feed_dict = {self.encoder_inputs[t]: input_seq_batched[t] for t in range(max_input_length)}
        feed_dict.update({self.labels[t]: target_seq_batched[t] for t in range(max_output_length)})

        feed_dict[self.keep_prob] = 0.5

        return feed_dict

    def make_inference_inputs(self, input_seq):
        '''This method is responsible for preparing the given sequences
           so that they can be used for inference using the model.'''
        max_input_length = self.cfg.get('max_input_length')

        input_seq_batched, _ = utils.batch(input_seq, max_sequence_length=max_input_length)

        feed_dict = {self.encoder_inputs[t]: input_seq_batched[t] for t in range(max_input_length)}
        feed_dict[self.keep_prob] = 1.0

        return feed_dict

class Model(object):
    '''Responsible for building the tensorflow graph, i.e.
       setting up the network so that it can be used by the
       Runner class. This implementation is heavily based on
       the tensorflow tutorial found at:
       https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/model_new.py'''

    # List of possible style of RNN cells
    CELL_FN = {
        # 'LSTM': rnn.LSTMCell,
        # 'GRU': rnn.GRUCell,
        # 'RNN': rnn.BasicRNNCell
    }
    
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
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.info('No embeddings given, initializing random embeddings')

            self.embeddings = tf.get_variable(
                name='embeddings',
                shape=[self.cfg.get('max_vocabulary_size'),
                       self.cfg.get('max_random_embeddings_size')],
                initializer=tf.random_uniform_initializer(-1, 1),
                dtype=tf.float32,
                trainable=True
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
                swap_memory=True,
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
                swap_memory=True,
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
            max_vocabulary_size = self.cfg.get('max_vocabulary_size')
            num_hidden_units = self.cfg.get('num_hidden_units')
            num_sampled = self.cfg.get('sampled_softmax_number_of_samples')
            
            w_t = tf.get_variable('out_proj_w', [max_vocabulary_size, num_hidden_units], dtype=tf.float32)
            w   = tf.transpose(w_t)
            b   = tf.get_variable('out_proj_b', [max_vocabulary_size], dtype=tf.float32)

            def sampled_softmax_loss(labels, inputs):
              labels = tf.reshape(labels, [-1, 1])

              # We need to compute the sampled_softmax_loss using 32bit floats to
              # avoid numerical instabilities.
              local_w_t = tf.cast(w_t, tf.float32)
              local_b = tf.cast(b, tf.float32)
              local_inputs = tf.cast(inputs, tf.float32)

              return tf.cast(
                  tf.nn.sampled_softmax_loss(
                      weights=local_w_t,
                      biases=local_b,
                      labels=labels,
                      inputs=local_inputs,
                      num_sampled=num_sampled,
                      num_classes=max_vocabulary_size))

            softmax_loss_fn = sampled_softmax_loss

        # Track the global step state when training
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights,
                                          softmax_loss_function=softmax_loss_fn)

        self.train_op = tf.train.AdadeltaOptimizer().minimize(self.loss, global_step=self.global_step)

    def __get_vocab_size(self):
        '''Returns the size of the vocabulary if the embeddings are
           already loaded, otherwise an error is thrown.'''
        embs = self.get_embeddings()

        if embs is None:
            raise Exception('no embeddings set via set_embeddings() before calling __get_vocab_size()')
        elif isinstance(embs, tf.Variable):
            return embs.get_shape()[0].value
        else:
            return embs.shape[0]
