#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""
This is some documentation
"""


def create_layer(prev, n, activation):
    """_summary_

    Args:
        prev (_type_): _description_
        n (_type_): _description_
        activation (_type_): _description_

    Returns:
        _type_: _description_
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation, kernel_initializer=initializer, name='layer')(prev)
    return layer