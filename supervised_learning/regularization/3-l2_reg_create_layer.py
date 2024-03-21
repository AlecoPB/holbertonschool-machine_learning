#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """_summary_

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_
        lambtha (_type_): _description_

    Returns:
        _type_: _description_
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # Apply L2 regularization to weights
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)

    # Create fully connected layer
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)
    
    return layer(prev)