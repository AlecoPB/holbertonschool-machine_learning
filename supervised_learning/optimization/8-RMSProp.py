#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the RMSProp optimization operation for a neural network.

    Args:
        loss (tf.Tensor): Loss of the network.
        alpha (float): Learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): Small number to avoid division by zero.

    Returns:
        tf.Operation: The RMSProp optimization operation.
    """
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                            decay=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)

    return train_op
