#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    Calculate the positional encoding for a transformer.
    """
    pe = np.zeros((max_seq_len, dm))

    # Create an array of position indices
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Create an array of dimension indices
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply the sine function to even indices in the array; 2i
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply the cosine function to odd indices in the array; 2i+1
    pe[:, 1::2] = np.cos(position * div_term)

    return pe


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate the scaled dot product attention.
    """
    # Calculate the dot product between Q and K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale matmul_qk by the square root of the dimension of the keys
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Calculate the attention weights using softmax
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Calculate the output by performing a weighted sum of the values
    output = tf.matmul(attention_weights, V)

    return output, attention_weights


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


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block class
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Call function
        """
        # Apply multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Apply the feed-forward network
        dense_hidden_output = self.dense_hidden(out1)
        dense_output = self.dense_output(dense_hidden_output)
        dense_output = self.dropout2(dense_output, training=training)
        out2 = self.layernorm2(out1 + dense_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """
    Decoder block class
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call function
        """
        # Apply the first multi-head attention (masked)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # Apply the second multi-head attention (with encoder output)
        attn2, attn_weights_block2 = self.mha2(out1, encoder_output,
                                               encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        # Apply the feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class
    """
    def __init__(self, N, dm, h, hidden,
                 input_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm,
                                    h,
                                    hidden,
                                    drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call function
        """
        seq_len = tf.shape(x)[1]

        # Generate the embeddings and add the positional encodings
        x = self.embedding(x)  # (batch, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        # Apply dropout to the positional encodings
        x = self.dropout(x, training=training)

        # Pass the input through each encoder block
        for block in self.blocks:
            x = block(x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder class
    """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm,
                                    h,
                                    hidden,
                                    drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call function
        """
        seq_len = tf.shape(x)[1]

        # Generate the embeddings and add the positional encodings
        x = self.embedding(x)  # (batch, target_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        # Apply dropout to the positional encodings
        x = self.dropout(x, training=training)

        # Pass the input through each decoder block
        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)

        return x


class Transformer(tf.keras.Model):
    """
    Transformer class
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Call function
        """
        # Pass the inputs through the encoder
        encoder_output = self.encoder(inputs, training, encoder_mask)

        # Pass the target and encoder output through the decoder
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)

        # Apply the final linear layer
        output = self.linear(decoder_output)

        return output
