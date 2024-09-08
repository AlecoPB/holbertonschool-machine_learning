#!/usr/bin/env python3
"""Module to perform forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    Parameters:
    rnn_cell (RNNCell): An instance of the RNNCell class.
    X (ndarray): The data to be used, shape (t, m, i).
    h_0 (ndarray): The initial hidden state, shape (m, h).

    Returns:
    H (ndarray): All hidden states, shape (t + 1, m, h).
    Y (ndarray): All outputs, shape (t, m, o).
    """
    t, m, i = X.shape  # t: number of time steps, m: batch size, i: input size
    h = h_0.shape[1]  # h: hidden state size
    o = rnn_cell.Wy.shape[1]  # o: output size

    # Initialize hidden states and outputs
    H = np.zeros((t + 1, m, h))  # (t+1, m, h), including initial hidden state
    Y = np.zeros((t, m, o))      # (t, m, o) for outputs at each time step

    # Set the initial hidden state
    H[0] = h_0

    # Perform forward propagation for each time step
    h_prev = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_next  # Store the next hidden state
        Y[step] = y           # Store the output for this step
        h_prev = h_next       # Update h_prev for the next step

    return H, Y
