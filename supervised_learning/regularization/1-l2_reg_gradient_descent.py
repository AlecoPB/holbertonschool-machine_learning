#!/usr/bin/env python3
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
    m = Y.shape[1]
    A = cache['A' + str(L)]

    # Calculate the gradient for the output layer (softmax)
    dZ = A - Y

    # Update weights and biases for the last layer
    weights['W' + str(L)] -= alpha * (np.dot(dZ, cache['A' + str(L - 1)].T)
                                      / m + (lambtha / m) * weights['W' + str(L)])
    weights['b' + str(L)] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m

    # Backpropagation for the hidden layers
    for l in range(L, 0, -1):
        A_prev = cache[f"A{l - 1}"]
        W = weights['W' + str(l + 1)]
        dA = np.dot(W.T, dZ)

        # Activation function derivative (assuming tanh here, modify as needed)
        dZ = dA * (1 - np.square(cache['A' + str(l)]))

        # Update weights and biases for the current layer
        weights['W' + str(l)] -= alpha * (np.dot(dZ, A_prev.T)
                                          / m + (lambtha / m) * weights['W' + str(l)])
        weights['b' + str(l)] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m
