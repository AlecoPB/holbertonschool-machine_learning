#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Parameters:
    Y (numpy.ndarray): One-hot encoded labels of shape (classes, m).
    weights (dict): A dictionary of the weights and biases of
    the neural network.
    cache (dict): A dictionary of the outputs and dropout masks
    of each layer of the neural network.
    alpha (float): The learning rate.
    keep_prob (float): The probability that a node will be kept.
    L (int): The number of layers of the network.
    """
    m = Y.shape[1]
    A = cache[f"A{L}"]

    # Calculate the gradient for the output layer (softmax)
    dZ = A - Y

    # Update weights and biases for the last layer
    weights[f"W{L}"] -= alpha * (np.dot(dZ, cache[f"A{L - 1}"].T) / m)
    weights[f"b{L}"] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m

    # Backpropagation for the hidden layers
    for l in range(L - 1, 0, -1):
        A_prev = cache[f"A{l - 1}"]
        W = weights[f"W{l}"]

        # Apply the dropout mask to the gradient
        dZ = (np.dot(W.T, dZ) * (1 - np.square(A_prev))
              * cache[f"dropout_mask{l}"] / keep_prob)

        # Update weights and biases for the current layer
        weights[f"W{l}"] -= alpha * (np.dot(dZ, A_prev.T) / m)
        weights[f"b{l}"] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m
