#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network with L2 regularization.

    Parameters:
    cost (tensor): A tensor containing the cost of the network without L2 regularization.
    model (tf.keras.Model): A Keras model with layers that include L2 regularization.

    Returns:
    tensor: A tensor containing the total cost including L2 regularization.
    """
    # Initialize the regularization cost
    reg_cost = 0

    # Iterate over the layers of the model
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            # Add the regularization contribution of the layer
            reg_cost += tf.reduce_sum(layer.kernel_regularizer(layer.kernel))

    # Total cost = base cost + regularization cost
    total_cost = cost + reg_cost
    return total_cost
