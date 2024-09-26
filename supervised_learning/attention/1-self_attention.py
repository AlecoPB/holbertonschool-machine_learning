#!/usr/bin/env python3
"""
RNNDecoder for machine translation.
"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder class that inherits from tensorflow.keras.layers.Layer
    for decoding machine translation sequences.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        Args:
            vocab: Integer representing the size of the output vocabulary.
            embedding: Integer representing the dimensionality of the embedding vector.
            units: Integer representing the number of hidden units in the RNN cell.
            batch: Integer representing the batch size.
        """
        super(RNNDecoder, self).__init__()

        # Public instance attributes
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

        # SelfAttention instance
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Public instance method that performs the forward pass of the RNNDecoder.
        """
        # Compute the context vector using self-attention mechanism
        context, _ = self.attention(s_prev, hidden_states)

        # Embed the input word x into an embedding vector
        x = self.embedding(x)

        # Concatenate the context vector and the embedding vector of x
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass concatenated vector through the GRU
        output, s = self.gru(x, initial_state=s_prev)

        # Reshape output for Dense layer processing
        output = tf.reshape(output, (-1, output.shape[2]))

        # Pass the output through the Dense layer to get the final word probabilities
        y = self.F(output)

        return y, s
