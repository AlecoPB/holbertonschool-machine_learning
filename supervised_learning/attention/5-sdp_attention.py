#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


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
