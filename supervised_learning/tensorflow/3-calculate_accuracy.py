#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Args:
        y (_type_): Placeholder for the labels of the input data
        y_pred (_type_): Tensor containing the network’s predictions
    Returns:
        A tensor containing the decimal accuracy of the prediction
    """
    # Calculate correct predictions
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
