#!/usr/bin/env python3
"""
This is some documenation
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in
    TensorFlow that includes L2 regularization.

    Parameters:
    prev (tf.Tensor): Tensor containing the output of the previous layer.
    n (int): The number of nodes the new layer should contain.
    activation (str): The activation function to be used on the layer.
    lambtha (float): The L2 regularization parameter.

    Returns:
    tf.Tensor: The output of the new layer.
    """
    # Create the layer with L2 regularization
    regularizer = tf.keras.regularizers.l2(lambtha)

    # Initialize weights
    weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                    mode=('fan_avg'))

    # Define the layer
    layer = tf.keras.layers.Dense(n,
                                  activation=activation,
                                  kernel_initializer=weights,
                                  kernel_regularizer=regularizer)(prev)

    return layer
