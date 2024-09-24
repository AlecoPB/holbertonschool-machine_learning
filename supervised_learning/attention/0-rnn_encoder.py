#!/usr/bin/env python3
"""
RNN encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RRN Encoder class
    """
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()

        # Public instance attributes
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # GRU layer with glorot_uniform initializer
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """"
        Initializes the hidden state to a tensor of zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Sequence of outputs and last hidden state
        """
        # Pass input through the embedding layer
        x = self.embedding(x)

        # Pass embedded input and initial state through the GRU layer
        outputs, hidden = self.gru(x, initial_state=initial)

        # Return both the full sequence of outputs and the last hidden state
        return outputs, hidden
