#!/usr/bin/env python3
"""Module to perform forward propagation for a deep RNN."""

import numpy as np

def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Parameters:
    rnn_cells (list): List of RNNCell instances of length l (number of layers).
    X (ndarray): Data input, shape (t, m, i).
    h_0 (ndarray): Initial hidden state for all layers, shape (l, m, h).

    Returns:
    H (ndarray): All hidden states, shape (t + 1, l, m, h).
    Y (ndarray): All outputs, shape (t, m, o).
    """
    t, m, i = X.shape  # t: number of time steps, m: batch size, i: input size
    l, _, h = h_0.shape  # l: number of layers, h: hidden state size

    # Initialize hidden states and outputs
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0  # Set the initial hidden state

    # Get output dimensionality from the last RNNCell's Wy matrix
    o = rnn_cells[-1].Wy.shape[1]
    Y = np.zeros((t, m, o))  # (t, m, o) for outputs at each time step

    # Forward propagation through each time step
    for step in range(t):
        x_t = X[step]  # Input at time step t
        h_prev = H[step]  # Hidden states at time step t for all layers

        for layer in range(l):
            rnn_cell = rnn_cells[layer]
            h_prev_layer = h_prev[layer]

            if layer == 0:
                h_next_layer, y = rnn_cell.forward(h_prev_layer, x_t)
            else:
                h_next_layer,
                y = rnn_cell.forward(h_prev_layer,
                                     H[step + 1][layer - 1])

            # Store the hidden state for the current layer
            H[step + 1][layer] = h_next_layer

        # Store the output from the last layer
        Y[step] = y

    return H, Y
