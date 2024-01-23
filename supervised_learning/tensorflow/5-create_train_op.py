#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Args:
        y (_type_): Placeholder for the labels of the input data
        y_pred (_type_): Tensor containing the network’s predictions
    Returns:
        A tensor containing the loss of the prediction
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
