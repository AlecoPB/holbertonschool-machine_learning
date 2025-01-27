#!/usr/bin/env python3
"""
Create masks
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Generates a padding mask for the input sequence.
    The mask is a tensor filled with 0s and 1s

    Args:
        seq: A tensor of shape `(batch_size, seq_len)` representing the
            input sequence.

    Returns:
        A mask tensor of shape `(batch_size, 1, 1, seq_len)`.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Generates a look-ahead mask for the target sequence.
    This mask prevents the decoder from attending to future tokens.

    Args:
        size: The size of the mask `(seq_len_out)`.

    Returns:
        A mask tensor of shape `(seq_len_out, seq_len_out)`.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Generates all necessary masks for training/validation.

    Args:
        inputs: A tf.Tensor of shape `(batch_size, seq_len_in)` representing
            the input sentence.
        target: A tf.Tensor of shape `(batch_size, seq_len_out)` representing
            the target sentence.
    """
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)

    # Look-ahead mask for the target sequence
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # Padding mask for the target sequence
    dec_target_padding_mask = create_padding_mask(target)

    # Combined mask for the first decoder attention block
    combined_mask = tf.maximum(look_ahead_mask, dec_target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
