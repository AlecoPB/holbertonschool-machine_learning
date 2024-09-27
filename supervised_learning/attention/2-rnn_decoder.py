#!/usr/bin/env python3
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()

        # Public instance attributes
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)  # Embedding layer for target vocabulary
        self.gru = tf.keras.layers.GRU(units, 
                              return_sequences=True, 
                              return_state=True,
                              recurrent_initializer='glorot_uniform')  # GRU layer with glorot_uniform initializer
        self.F = tf.keras.layers.Dense(vocab)  # Dense layer for output, mapping to vocab size

        # Instantiate SelfAttention layer
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # Step 1: Calculate the context vector using the attention mechanism
        context, _ = self.attention(s_prev, hidden_states)  # Context from attention layer

        # Step 2: Embed the input word x (shape: (batch, 1)) into a dense vector
        x = self.embedding(x)  # Shape: (batch, 1, embedding_dim)

        # Step 3: Concatenate the context vector with the embedded input word
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)  # Shape: (batch, 1, embedding_dim + units)

        # Step 4: Pass the concatenated input through the GRU layer
        output, s = self.gru(x, initial_state=s_prev)  # Output shape: (batch, 1, units), s shape: (batch, units)

        # Step 5: Pass the GRU output through the Dense layer to get the final predicted word (logits)
        y = self.F(output)  # Shape: (batch, 1, vocab)

        # Step 6: Remove the sequence dimension (1) from the output y
        y = tf.squeeze(y, axis=1)  # Shape: (batch, vocab)

        return y, s

# Example usage
# vocab_size = 10000, embedding_dim = 256, units = 512, batch_size = 64
decoder = RNNDecoder(vocab=10000, embedding=256, units=512, batch=64)

# x is the previous word in target sequence, shape: (batch_size, 1)
x = tf.random.uniform((64, 1), maxval=10000, dtype=tf.int32)  # Random target sequence word indices

# s_prev is the previous hidden state, shape: (batch_size, units)
s_prev = tf.random.uniform((64, 512))  # Random decoder hidden state

# hidden_states is the encoder outputs, shape: (batch_size, input_seq_len, units)
hidden_states = tf.random.uniform((64, 10, 512))  # Random encoder outputs

# Get decoder output
y, s = decoder(x, s_prev, hidden_states)

print("Output shape:", y.shape)  # Expected: (64, 10000) -> one-hot logits over vocab
print("New state shape:", s.shape)  # Expected: (64, 512) -> new hidden state
