#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import tensorflow as tf


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
    # Create a dense layer with L2 regularization
    layer = tf.keras.layers.Dense(
        units=n,  # Number of nodes
        activation=activation,  # Activation function
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    )
    
    output = layer(prev)
    # layer = tf.keras.layers.Dense(3, kernel_regularizer='l2')
    
    return output