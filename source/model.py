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
    # List of possible style of RNN cells
    CELL_FN = {
        'LSTM': rnn.BasicLSTMCell,
        'GRU': rnn.GRUCell,
        'RNN': rnn.BasicRNNCell
    }

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        self.__build_model()
        self.__output_vars = None

    def __build_model(self):
        num_layers = self.cfg.get('num_encoder_layers') + self.cfg.get('num_decoder_layers')
        max_inp_len = self.cfg.get('max_input_length')
        max_out_len = self.cfg.get('max_output_length')
        hidden_units = self.cfg.get('num_hidden_units')
        vocab_len = len(self.cfg.get('vocabulary_dict'))
        embeddings_m = self.cfg.get('embeddings_matrix')
        embeddings_size = self.cfg.get('max_random_embeddings_size')
        buckets = self.cfg.get('buckets')
        num_samples = self.cfg.get('sampled_softmax_number_of_samples')
        cell_type = self.cfg.get('cell_type')
        dtype = tf.float32

        # TODO: Make configurable!
        learning_rate = 0.001
        learning_rate_decay_factor = 0.99
        max_gradient_norm = 10.0

        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        self.output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < vocab_len:
            w_t = tf.get_variable('proj_w', [vocab_len, hidden_units], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable('proj_b', [vocab_len], dtype=dtype)
            self.output_projection = (w, b)
  
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

        cell_class = self.CELL_FN[cell_type]

        def wrap_dropout(cell):
            return rnn.DropoutWrapper(cell, input_keep_prob=self.cfg.get('dropout_input_keep_prob'),
                                            output_keep_prob=self.cfg.get('dropout_output_keep_prob'))

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            cell_obj = cell_class(hidden_units)

            if self.cfg.get('train'):
                cell_obj = wrap_dropout(cell_obj)

            return cell_obj

        cell = None
        
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        else:
            cell = single_cell()

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=vocab_len,
                num_decoder_symbols=vocab_len,
                embedding_size=hidden_units,
                output_projection=self.output_projection,
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
            if self.output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()

        self.gradient_norms = []
        self.updates = []

        # Apply gradient clipping if we're in training mode
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

        if len(input_seq) < batch_size:
          batch_size = len(input_seq)

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
                padding_parts = [Config.PAD_WORD_IDX for i in range(max_output_length - len(current_target_seq))]
                current_target_seq += padding_parts
            
            current_target_seq.insert(0, Config.GO_WORD_IDX)

            if len(current_target_seq) > max_out_len:
                set_eos_word = current_target_seq[-1] == Config.EOS_WORD_IDX
                current_target_seq = current_target_seq[:max_out_len]

                if set_eos_word:
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
        buckets = self.cfg.get('buckets')

        if self.__output_vars is None and self.output_projection is not None:
            self.__output_vars = self.outputs

            # Only restore when we're doing inference while training, otherwise
            # the restore of the logits is done when constructing the graph
            if self.cfg.get('train'):
                for b in range(len(buckets)):
                  self.__output_vars[b] = [
                      tf.matmul(output, self.output_projection[0]) + self.output_projection[1]
                      for output in self.__output_vars[b]
                  ]

        for i in range(decoder_size):
            outputs_list.append(self.__output_vars[bucket_id][i])

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


