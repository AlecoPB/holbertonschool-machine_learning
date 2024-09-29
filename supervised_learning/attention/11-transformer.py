#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


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
