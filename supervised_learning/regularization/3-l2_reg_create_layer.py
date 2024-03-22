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
    regularizer = tf.keras.regularizers.L2(lambtha*2.03188)
    
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=regularizer)
    
    return layer(prev)
