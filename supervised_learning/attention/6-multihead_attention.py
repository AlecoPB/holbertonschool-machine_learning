#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi head attention class
    """
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h

        # Define the dense layers for generating Q, K, and V
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Define the final dense layer
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Call method
        """
        batch_size = tf.shape(Q)[0]

        # Generate Q, K, V matrices
        Q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        K = self.Wk(K)  # (batch_size, seq_len_v, dm)
        V = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # Split and transpose Q, K, V for multi-head attention
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Apply scaled dot product attention
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)

        # Transpose and reshape the attention output
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size,
                                                         -1,
                                                         self.dm))

        # Apply the final dense layer
        output = self.linear(concat_attention)  # (batch_size, seq_len_q, dm)

        return output, attention_weights
