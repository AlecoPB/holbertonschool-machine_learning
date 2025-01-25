#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with Dropout regularization.

    Parameters:
    X (numpy.ndarray): Input data for the network with shape (nx, m),
    where nx is the number of input features and m is the number of data points.
    weights (dict): A dictionary containing the weights and biases of the neural network.
    L (int): The total number of layers in the network.
    keep_prob (float): The probability that a neuron will be retained.

    Returns:
    dict: A dictionary containing the outputs of each layer and the dropout masks used.
    """
    # Initialize the cache with the input for the first layer
    cache = {'A0': X}

    for i in range(1, L + 1):
        # Get the activation output from the previous layer
        prev_A = cache[f"A{i - 1}"]
        Z = np.dot(weights[f"W{i}"], prev_A) + weights[f"b{i}"]

        # Apply tanh activation and dropout for all layers except the last
        if i < L:
            A = np.tanh(Z)
            # Generate a dropout mask and store it in the cache
            D = np.random.binomial(1, keep_prob, size=A.shape)
            cache[f"D{i}"] = D
            # Apply the dropout mask and scale the activations
            A *= D
            A /= keep_prob
        else:
            # For the output layer, use softmax activation without dropout
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        # Save the activation output in the cache
        cache[f"A{i}"] = A

    # Return the cache containing all activations and masks
    return cache