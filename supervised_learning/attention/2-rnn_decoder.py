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
        # Apply the embedding layer to the input x
        x = self.embedding(x)

        # Apply self-attention to get the context vector
        context_vector, _ = self.attention(s_prev, hidden_states)

        # Concatenate the context vector with x
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pass the concatenated vector to the GRU
        output, state = self.gru(x, initial_state=s_prev)

        # Pass the GRU output through the dense layer to get the final output
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, state
