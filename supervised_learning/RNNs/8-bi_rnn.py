#!/usr/bin/env python3
"""Module containing the BidirectionalCell class."""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell."""

    def __init__(self, i, h, o):
        """Initialize the BidirectionalCell.

        Parameters:
        i (int): Dimensionality of the data.
        h (int): Dimensionality of the hidden states.
        o (int): Dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(i + h, h))  # Forward hidden weights
        self.Whb = np.random.normal(size=(i + h, h))  # Backward hidden weights
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))  # Forward hidden bias
        self.bhb = np.zeros((1, h))  # Backward hidden bias
        self.by = np.zeros((1, o))   # Output bias

    def forward(self, h_prev, x_t):
        """
        Calculate the next hidden state in the forward direction.

        Parameters:
        h_prev (ndarray): Previous hidden state, shape (m, h).
        x_t (ndarray): Data input for the cell, shape (m, i).

        Returns:
        h_next (ndarray): Next hidden state in the forward direction.
        """
        x_h_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(x_h_concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculate the previous hidden state in the backward direction.

        Parameters:
        h_next (ndarray): Next hidden state, shape (m, h).
        x_t (ndarray): Data input for the cell, shape (m, i).

        Returns:
        h_prev (ndarray): Previous hidden state in the backward direction.
        """
        x_h_concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(x_h_concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculate all outputs for the RNN.

        Parameters:
        H (ndarray): Concatenated hidden states from both directions,
                     shape (t, m, 2 * h).

        Returns:
        Y (ndarray): The outputs for each time step, shape (t, m, o).
        """
        # Apply the output layer to all time steps
        t, m, _ = H.shape
        Y = H @ self.Wy + self.by  # Linear transformation
        # Apply softmax to output the probabilities for each time step
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2, keepdims=True)

        return Y

def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    Parameters:
    X (ndarray): Data input, shape (t, m, i).
    h_0 (ndarray): Initial hidden state in the forward direction, shape (m, h).
    h_t (ndarray): Initial hidden state in the backward direction, shape (m, h)

    Returns:
    H (ndarray): All concatenated hidden states, shape (t, m, 2 * h).
    Y (ndarray): All outputs, shape (t, m, o).
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize arrays to store hidden states
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    # Forward direction
    h_prev = h_0
    for step in range(t):
        h_next = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_next
        h_prev = h_next

    # Backward direction
    h_next = h_t
    for step in reversed(range(t)):
        h_prev = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_prev
        h_next = h_prev

    # Concatenate forward and backward hidden states at each time step
    H = np.concatenate((H_forward, H_backward), axis=2)

    # Calculate the outputs
    Y = bi_cell.output(H)

    return H, Y
