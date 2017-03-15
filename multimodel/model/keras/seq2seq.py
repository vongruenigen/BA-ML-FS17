#
# BA ML FS17 - Dirk von Grünigen & Martin Weilenmann
#
# Description: This modul contains the base model which
#              all models should inherit from. Also contains
#              some register/factory functions for models.
#

from seq2seq.models import AttentionSeq2Seq, SimpleSeq2Seq

from multimodel.model.keras.base import Base

class Seq2Seq(Base):
    '''This class represents a sequence-to-sequence model and uses
       the keras library found at: https://github.com/farizrahman4u/seq2seq'''

    def __init__(self, cfg):
        '''Constructor for the Seq2Seq class.'''
        super(Seq2Seq, self).__init__(cfg)

    def build(self):
        '''This function is responsible for initializing and building the model.'''
        self.model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8, depth=3)
        self.model.compile(loss='mse', optimizer='rmsprop')

    def get_model(self):
        '''Returns the model. An error is thrown if build() is not called calling
           this function.'''
        pass
