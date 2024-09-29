#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder
    """
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()

        # Public instance attributes
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(units=vocab)

        # Instantiate SelfAttention layer
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Call function
        """
        # Step 1: Calculate the context vector using attention
        context, _ = self.attention(s_prev, hidden_states)

        # Step 2: Embed the input word x into a dense vector
        x = self.embedding(x)  # Shape: (batch, 1, embedding_dim)

        # Step 3: Concatenate the context vector with input word
        x = tf.concat([context, x], axis=-1)

        # Step 4: Pass the concatenated input through the GRU layer
        output, s = self.gru(x, initial_state=s_prev)

        y = self.F(output)  # Shape: (batch, 1, vocab)

        return y, s
