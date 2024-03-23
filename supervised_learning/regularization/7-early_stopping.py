#!/usr/bin/env python3
import numpy as np

def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: numpy.ndarray - tensor containing the output of the previous layer
        n: int - number of nodes the new layer should contain
        activation: str - activation function to be used on the layer
        keep_prob: float - probability that a node will be kept

    Returns:
        numpy.ndarray: the output of the new layer
    """
    # Initialize weights and biases for the new layer
    W = np.random.randn(n, prev.shape[0]) * np.sqrt(2 / prev.shape[0])  # Using He initialization
    b = np.zeros((n, 1))

    # Linear transformation
    Z = np.dot(W, prev) + b

    # Activation function
    if activation == "tanh":
        A = np.tanh(Z)
    elif activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
    elif activation == "relu":
        A = np.maximum(0, Z)
    else:
        raise ValueError("Unsupported activation function.")

    # Dropout
    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
    A = np.multiply(A, D)
    A /= keep_prob

    return A
