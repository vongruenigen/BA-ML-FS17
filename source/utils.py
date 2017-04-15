#
# BA ML FS17 - Dirk von Gruenigen & Martin Weilenmann
#
# Description: This module contains several utility methods.
#

import numpy as np
import pickle
import itertools

from os import path
from config import Config

def load_vocabulary(path):
    '''Loads the vocabulary at the given path. The specification the
       vocabulary has to obey can be found in the description of the
       params in the Config class.'''
    with open(path, 'rb') as f:
        return pickle.load(f)

def reverse_vocabulary(vocabulary):
    '''Reverses the vocabulary in the sense, that the keys of the new
       one will be the indices and the values are the words. This is
       used to convert a list of indices to the respective sentence.'''
    return {v: k for k, v in vocabulary.items()}

def prepare_embeddings_and_vocabulary(embeddings, vocabulary):
    '''Adds an embeddings for an unknown word at the top of the embeddings
       matrix and updates the vocabulary accordingly.'''
    unknown_embedding = np.random.uniform(-1.0, 1.0, size=(1, embeddings.shape[1]))
    embeddings = np.vstack([unknown_embedding, embeddings])

    # NOTE: The +3 when setting the index comes from the fact
    #       that the embedding for the unknown, eos and pad word
    #       will be inserted as the first rows of the embeddings
    #       matrix in order to keep it simple.
    new_vocabulary = {w: idx+4 for w, idx in vocabulary.items()}

    new_vocabulary[Config.UNKNOWN_WORD_TOKEN] = Config.UNKNOWN_WORD_IDX
    new_vocabulary[Config.PAD_WORD_TOKEN] = Config.PAD_WORD_IDX
    new_vocabulary[Config.EOS_WORD_TOKEN] = Config.EOS_WORD_IDX
    new_vocabulary[Config.GO_WORD_TOKEN] = Config.GO_WORD_IDX

    return embeddings.astype('float32'), new_vocabulary

def batch(inputs, max_sequence_length=None, time_major=True):
    '''Taks a list of inputs and converts them into a data batch which
       can be used by the model for training or inference.'''
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    if time_major:
      # [batch_size, max_time] -> [max_time, batch_size]
      inputs_time_major = inputs_batch_major.swapaxes(0, 1)
      return inputs_time_major, sequence_lengths
    else:
      return inputs_batch_major, sequence_lengths

def flatten_list(list_to_flatten):
    '''Returns a flattened version of the list supplied as a parameter.'''
    return list(itertools.chain(*list_to_flatten))

def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    '''Generates batches of random integer sequences.'''
    if length_from > length_to:
            raise ValueError('length_from > length_to')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    while True:
        yield [
            np.random.randint(low=vocab_lower,
                              high=vocab_upper,
                              size=random_length()).tolist()
            for _ in range(batch_size)
        ]
