#!/usr/bin/env python3
"""Module containing the GRUCell class."""

import numpy as np


def sigmoid(x):
    """Applies the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell."""

    def __init__(self, i, h, o):
        """Initialize the GRUCell.

        Parameters:
        i (int): Dimensionality of the data.
        h (int): Dimensionality of the hidden state.
        o (int): Dimensionality of the outputs.
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        Parameters:
        h_prev (ndarray): Previous hidden state, shape (m, h).
        x_t (ndarray): Data input for the cell, shape (m, i).

        Returns:
        h_next (ndarray): Next hidden state, shape (m, h).
        y (ndarray): Output of the cell, shape (m, o).
        """
        # Concatenate h_prev and x_t for gate calculations
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = sigmoid(h_x @ self.Wz + self.bz)

        # Reset gate
        r_t = sigmoid(h_x @ self.Wr + self.br)

        # Intermediate hidden state candidate
        h_x_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h̃_t = np.tanh(h_x_reset @ self.Wh + self.bh)

        # Next hidden state
        h_next = z_t * h_prev + (1 - z_t) * h̃_t

        # Output using softmax
        y_linear = h_next @ self.Wy + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y
