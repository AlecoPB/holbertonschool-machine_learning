#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Args:
        y (_type_): Placeholder for the labels of the input data
        y_pred (_type_): Tensor containing the network’s predictions
    Returns:
        A tensor containing the loss of the prediction
    """
    # Compute softmax cross-entropy loss

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
        labels=y, logits=y_pred))
    return loss
