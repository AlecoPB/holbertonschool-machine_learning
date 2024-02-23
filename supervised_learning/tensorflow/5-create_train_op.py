#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Args:
        loss (_type_): Tensor containing the loss of the prediction
        alpha (_type_): Learning rate for the gradient descent optimizer
    Returns:
        A tensor containing the operation to minimize the loss
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
