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

from os import path

from multimodel.constants import RESULTS_PATH
from multimodel.model.keras.seq2seq import Seq2Seq
from multimodel.runner.base import Base
from multimodel.config import Config
from multimodel.data_loader.conversational import ConversationalDataLoader

class KerasRunner(Base):
    '''This class is responsible for the training,
       testing and running of the model.'''

    def __init__(self, cfg_path):
        '''Creates a new instance of the KerasRunner class.'''
        super(KerasRunner, self).__init__(cfg_path)

    def train(self):
        '''This method is responsible for training a model
           with the settings defined in the config.'''
        model = self.__create_model()
        training_batches = None
        test_batches = None

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

        batch_data_x, batch_data_y = self.prepare_data_batch(training_batches)

        import pdb
        pdb.set_trace()

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

    def __store_model(self, session, model, epoch_nr):
        '''Stores the given model in the current results directory.
           This model can then later be reloaded with the __load_model()
           method.'''
        raise Exception('to be implemented')

    def __create_model(self):
        '''Creates a model based on the name given.'''
        self.model = Seq2Seq(self.cfg)
        self.model.build()
