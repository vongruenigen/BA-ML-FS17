#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This module contains several utility methods.
#

import numpy as np

from gensim.models import Word2Vec
from data_loader import DataLoader

def __add_unknown_word_embedding(embs):
    '''Helper function which adds a column for the embedding
       of the unknown word.'''
    embedding_for_unknown = np.random.uniform(-1.0, 1.0, size=(1, embs.shape[1]))
    return np.vstack([embs, embedding_for_unknown])

def load_w2v_embeddings(path):
    '''Loads the word2vec embeddings at the given path. It returns
       the embedding matrix as a numpy ndarray and the vocabulary as
       a dict object.'''
    w2v_model = Word2Vec.load(path)

    embeddings = w2v_model.syn0
    embeddings = __add_unknown_word_embedding(embeddings)

    vocab = {k: w.index for k, w in w2v_model.vocab.items()}
    vocab['UNKNOWN'] = embeddings.shape[0] - 1

    return embeddings, vocab

def load_ft_embeddings(path):
    '''Loads the fastTrack embeddings at the given path. It returns
       the embedding matrix as a numpy ndarray and the vocabulary as
       a dict object.'''
    raise Exception('Loading of ft embeddings is not implemented yet!')

def batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

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