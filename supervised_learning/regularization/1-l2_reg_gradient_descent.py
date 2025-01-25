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
    A = cache[f"A{L}"]

    # Calculate the gradient for the output layer (softmax)
    dZ = A - Y

    # Update weights and biases for the last layer
    weights['W' + str(L)] -= alpha * (np.dot(dZ, cache['A' + str(L - 1)].T)
                                      / m + (lambtha / m) * weights['W' + str(L)])
    weights['b' + str(L)] -= alpha * np.sum(dZ, axis=1, keepdims=True) / m

    # Backpropagation for the hidden layers
    for l in range(L, 0, -1):
        A_prev = cache[f"A{l - 1}"]

        # L2 regularization term
        l2_reg = (lambtha / m) * weights[f"W{l}"]

        # Gradient of loss with respect to weights and biases
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + l2_reg
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if l > 1:
            # Gradient of the activation function (tanh)
            dZ = np.matmul(weights[f"W{l}"].T, dZ) * (1 - np.square(A_prev))

        # Updating weights and biases
        weights[f"W{l}"] -= alpha * dW
        weights[f"b{l}"] -= alpha * db