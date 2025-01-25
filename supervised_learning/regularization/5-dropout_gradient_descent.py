#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache,
                             alpha, keep_prob, L):
    """
    Adjusts the weights of a neural network using gradient descent
    with Dropout regularization.

    Parameters:
    Y (numpy.ndarray): One-hot encoded array of shape
    (classes, m) containing
    the correct labels for the dataset.
    weights (dict): A dictionary holding the weights and biases
    of the neural network.
    cache (dict): A dictionary containing the outputs and dropout
    masks for each
    layer of the neural network.
    alpha (float): The learning rate for weight updates.
    keep_prob (float): The probability that a neuron will be
    retained.
    L (int): The total number of layers in the network.

    Returns:
    None: The weights dictionary is updated in place.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    # Iterate through layers in reverse order:
    for i in range(L, 0, -1):
        # Get the activation output from the previous layer
        prev_A = cache[f"A{i - 1}"]

        # Calculate the gradient of the loss with respect to weights and biases
        dW = (1 / m) * np.dot(dZ, prev_A.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            # Compute the gradient of the activation function (tanh)
            dA = np.dot(weights[f"W{i}"].T, dZ)
            # Apply the dropout mask from the cache
            dA *= cache[f"D{i - 1}"]
            # Scale the activation gradients by the keep probability
            dA /= keep_prob
            # Apply the tanh gradient
            dZ = dA * (1 - np.square(cache[f"A{i - 1}"]))

        # Update the weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
