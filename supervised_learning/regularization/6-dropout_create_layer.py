#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Constructs a dense layer with dropout regularization.

    Parameters:
    - prev: Tensor containing the output from the previous layer.
    - n: The number of nodes in the new layer.
    - activation: The activation function to be applied in the new layer.
    - keep_prob: The probability that a neuron will be retained.
    - training: A boolean indicating if the model is currently in training mode.

    Returns: The output tensor of the new layer.
    """
    # Initialize weights using He et al. method
    init_weights = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg")

    # Create the dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init_weights
    )

    # Pass the previous layer's output through the dense layer
    output = layer(prev)

    # Apply dropout only if the model is in training mode
    if training:
        dropout = tf.keras.layers.Dropout(rate=(1 - keep_prob))
        output = dropout(output, training=training)

    return output
