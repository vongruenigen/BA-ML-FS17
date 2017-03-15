#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This modul contains the base model which
#              all models should inherit from. Also contains
#              some register/factory functions for models.
#

import tensorflow as tf

class Base(object):
    '''Base class which all classes should implement to be used as a
       model. There are several functions which must be implemented by
       the deriving class:

       * build():                  This function receives a tensorflow session and is
                                   expected to build the graph which reflects the model
                                   to be used.

       * get_metric_ops():         Should return a dict where the keys are the
                                   names of the metric ops and the values should
                                   be the ops itself.

       * get_train_op():           Should return the op used for training the model.
       * get_inference_op():       Should return the op which can be used to do
                                   inference with a trained or loaded model.

       * make_train_inputs():      This function is expected to puts batches of
                                   data in feed dict which can be used fot training.

       * make_inference_ inputs(): This function is expected to puts batches of
                                   data in feed dict which can be used fot training.

    '''

    def __init__(self, cfg, session):
        '''Initializes the model with the given configuration.'''
        self.cfg = cfg
        self.session = session
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def get_global_step(self):
        '''Returns the variable used for tracking the global step.'''
        return self.global_step

    def build(self):
        '''Stub function for build the model. This function must
           be overriden by deriving classes and should build the
           model using the tensorflow session set in the constructor.
           Raises an exception in case it is not overriden or called
           on a Base object.'''
        raise Exception('must be implemented by the deriving class!')

    def get_metric_ops(self):
        '''Stub function for return the ops used for calculating metrics.'''
        raise Exception('must be implemented by the deriving class!')

    def get_train_op(self):
        '''Stub function for return the ops used for training the model.'''
        raise Exception('must be implemented by the deriving class!')

    def get_inference_op(self):
        '''Stub function for return the ops used for doing inference on the model.'''
        raise Exception('must be implemented by the deriving class!')

    def make_train_inputs(self, inpute_seqs, output_seqs):
        '''Stub function for prepare data for feeding it into the model for training.'''
        raise Exception('must be implemented by the deriving class!')

    def make_inference_inputs(self, inputs_seqs):
        '''Stub function for prepare data for feeding it into the model for inference.'''
        raise Exception('must be implemented by the deriving class!')


    def build(self):
        '''Stub function for build the model. This function must
           be overriden by deriving classes and should build the
           model using the tensorflow session set in the constructor.
           Raises an exception in case it is not overriden or called
           on a Base object.'''
        raise Exception('must be implemented by the deriving class!')

    def get_metric_ops(self):
        '''Stub function for return the ops used for calculating metrics.'''
        raise Exception('must be implemented by the deriving class!')

    def get_train_op(self):
        '''Stub function for return the ops used for training the model.'''
        raise Exception('must be implemented by the deriving class!')

    def get_inference_op(self):
        '''Stub function for return the ops used for doing inference on the model.'''
        raise Exception('must be implemented by the deriving class!')

    def make_train_inputs(self, inpute_seqs, output_seqs):
        '''Stub function for prepare data for feeding it into the model for training.'''
        raise Exception('must be implemented by the deriving class!')

    def make_inference_inputs(self, inputs_seqs):
        '''Stub function for prepare data for feeding it into the model for inference.'''
        raise Exception('must be implemented by the deriving class!')
