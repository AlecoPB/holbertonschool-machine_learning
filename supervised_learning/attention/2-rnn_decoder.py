#!/usr/bin/env python3
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()

        # Public instance attributes
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, 
                              return_sequences=True, 
                              return_state=True,
                              recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

        # Instantiate SelfAttention layer
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # Step 1: Calculate the context vector using the attention mechanism
        context, _ = self.attention(s_prev, hidden_states)

        # Step 2: Embed the input word x (shape: (batch, 1)) into a dense vector
        x = self.embedding(x)  # Shape: (batch, 1, embedding_dim)

        # Step 3: Concatenate the context vector with the embedded input word
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Step 4: Pass the concatenated input through the GRU layer
        output, s = self.gru(x, initial_state=s_prev)
