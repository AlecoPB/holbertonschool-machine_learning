#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.

    Parameters:
    prev (tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function that should
    be used on the output of the layer.

    Returns:
    tensor: The activated output for the layer.
    """
    # Create a dense layer
    dense =\
        tf.keras.layers.Dense(units=n,
                              kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'))

    # Apply the dense layer to the inputs
    Z = dense(prev)

    # Calculate mean and variance
    batch_mean, batch_var = tf.nn.moments(Z, [0])

    # Create two trainable parameters, gamma and beta
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    # Implement batch normalization
    BN = tf.nn.batch_normalization(Z, batch_mean, batch_var, beta, gamma, 1e-8)

    # Apply the activation function
    A = activation(BN)

    return A
