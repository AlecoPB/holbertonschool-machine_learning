#!/usr/bin/env python3
"""Module containing the RNNCell class."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initialize the RNNCell.

        Parameters:
        i (int): Dimensionality of the data.
        h (int): Dimensionality of the hidden state.
        o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Parameters:
        h_prev (ndarray): Previous hidden state, shape (m, h).
        x_t (ndarray): Data input for the cell, shape (m, i).

        Returns:
        h_next (ndarray): Next hidden state, shape (m, h).
        y (ndarray): Output of the cell, shape (m, o).
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(h_x @ self.Wh + self.bh)
        y_linear = h_next @ self.Wy + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)
        return h_next, y
