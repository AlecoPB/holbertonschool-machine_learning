#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation.

    Args:
        loss: The loss of the network.
        alpha: The learning rate.
        beta2: The RMSProp weight.
        epsilon: A small number to avoid division by zero.

    Returns:
        The RMSProp optimization operation.
    """
    optimizer = tf.train.RMSProp(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
