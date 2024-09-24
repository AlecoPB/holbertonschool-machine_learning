#!/usr/bin/env python3
"""
Some documentation
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self attention class
    """
    def __init__(self, units):
        super(SelfAttention, self).__init__()

        # Public instance attributes (Dense layers)
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Call method that returns context vector and attention weights
        """
        # Expand the s_prev shape to match the hidden states for broadcasting
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        # Apply W to s_prev and U to hidden_states
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded
                                         ) + self.U(hidden_states)))

        # Calculate attention weights with softmax
        weights = tf.nn.softmax(score, axis=1)

        # Compute context vector as the weighted sum of hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
