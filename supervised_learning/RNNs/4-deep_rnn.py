#!/usr/bin/env python3
"""Module for deep RNN forward propagation."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    Parameters:
    rnn_cells (list): List of RNNCell instances, one per layer.
    X (ndarray): Data to be used, shape (t, m, i).
    h_0 (ndarray): Initial hidden states, shape (l, m, h).

    Returns:
    H (ndarray): All hidden states, shape (l, t + 1, m, h).
    Y (ndarray): All outputs, shape (t, m, o).
    """
    t, m, i = X.shape  # t: time steps, m: batch size, i: input size
    l, _, h = h_0.shape  # l: number of layers, h: hidden state size
    o = rnn_cells[-1].Wy.shape[1]  # Output size from the last RNN cell

    # Initialize H to store hidden states for all layers and all time steps
    H = np.zeros((l, t + 1, m, h))
    H[:, 0, :, :] = h_0  # Set initial hidden states

    # Initialize Y to store the outputs
    Y = np.zeros((t, m, o))

    # Perform forward propagation through each time step
    for step in range(t):
        x_input = X[step]  # Input for the first layer is X at time step `step`

        # Iterate over each layer in the RNN
        for layer in range(l):
            h_prev = H[layer, step]  # Get the previous hidden state for this layer
            rnn_cell = rnn_cells[layer]

            # Forward pass for the current layer
            h_next, y = rnn_cell.forward(h_prev, x_input)

            # Store the next hidden state for this layer
            H[layer, step + 1] = h_next

            # The input for the next layer is the hidden state of the current layer
            x_input = h_next

        # The output from the last layer is stored in Y
        Y[step] = y

    return H, Y
