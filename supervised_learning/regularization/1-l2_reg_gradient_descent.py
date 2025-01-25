#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural
    network using gradient descent with L2 regularization.

    Parameters:
    Y (numpy.ndarray): One-hot encoded labels of shape (classes, m).
    weights (dict): A dictionary of the weights and biases
    of the neural network (e.g., W1, b1, ..., WL, bL).
    cache (dict): A dictionary of the outputs of each
    layer of the neural network (e.g., A1, ..., AL).
    alpha (float): The learning rate.
    lambtha (float): The L2 regularization parameter.
    L (int): The number of layers of the network.
    """
    m = Y.shape[1]  # Number of data points
    dZ = cache[f"A{L}"] - Y  # Compute the difference for the output layer

    # Iterate through layers in reverse order:
    for i in range(L, 0, -1):
        # Get the activation output from the previous layer
        prev_A = cache[f"A{i - 1}"]

        # Calculate the L2 regularization term for the current layer's weights
        l2_reg = (lambtha / m) * weights[f"W{i}"]

        # Compute the gradients for weights and biases
        dW = (1 / m) * np.dot(dZ, prev_A.T) + l2_reg
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Calculate the gradient of the activation function (tanh)
            dZ = np.dot(weights[f"W{i}"].T, dZ) * (1 - np.square(prev_A))

        # Update the weights and biases in place
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
