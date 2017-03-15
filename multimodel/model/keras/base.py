#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This modul contains the base model which
#              all models should inherit from. Also contains
#              some register/factory functions for models.
#

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

    def __init__(self, cfg):
        '''Initializes the model with the given configuration.'''
        self.cfg = cfg
