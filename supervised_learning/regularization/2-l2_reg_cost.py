#!/usr/bin/env python3
"""
This is some documentation
"""
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
    return cost + model.losses