#!/usr/bin/env python3
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        
        # Public instance attributes (Dense layers)
        self.W = tf.keras.layers.Dense(units)  # To apply to the previous decoder hidden state
        self.U = tf.keras.layers.Dense(units)  # To apply to the encoder hidden states
        self.V = tf.keras.layers.Dense(1)      # To apply to the tanh of W and U outputs

    def call(self, s_prev, hidden_states):
        # Expand the s_prev shape to match the hidden states for broadcasting
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)  # Shape: (batch, 1, units)

        # Apply W to s_prev and U to hidden_states
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))  # Shape: (batch, input_seq_len, 1)

        # Calculate attention weights with softmax
        weights = tf.nn.softmax(score, axis=1)  # Shape: (batch, input_seq_len, 1)

        # Compute context vector as the weighted sum of hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # Shape: (batch, units)

        return context, weights
