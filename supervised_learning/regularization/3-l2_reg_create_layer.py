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
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.refularizers.L2(lambtha)
    )
    
    output = layer(prev)
    
    return output