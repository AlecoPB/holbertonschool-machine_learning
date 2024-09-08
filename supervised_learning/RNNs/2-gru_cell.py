#!/usr/bin/env python3
"""Module containing the GRUCell class."""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit (GRU) cell."""

    def __init__(self, i, h, o):
        """Initialize the GRUCell.

        Parameters:
        i (int): Dimensionality of the data.
        h (int): Dimensionality of the hidden state.
        o (int): Dimensionality of the outputs.
        """
        self.Wz = np.random.normal(size=(i + h, h))  # Update gate weights
        self.Wr = np.random.normal(size=(i + h, h))  # Reset gate weights
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))      # Output weights

        self.bz = np.zeros((1, h))  # Update gate biases
        self.br = np.zeros((1, h))  # Reset gate biases
        self.bh = np.zeros((1, h))  # Intermediate hidden state biases
        self.by = np.zeros((1, o))  # Output biases

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Parameters:
        h_prev (ndarray): Previous hidden state, shape (m, h).
        x_t (ndarray): Data input for the cell, shape (m, i).

        Returns:
        h_next (ndarray): Next hidden state, shape (m, h).
        y (ndarray): Output of the cell, shape (m, o).
        """
        m, h = h_prev.shape
        x_h_concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(x_h_concat @ self.Wz + self.bz)

        # Reset gate
        r_t = self.sigmoid(x_h_concat @ self.Wr + self.br)

        # Intermediate hidden state
        h_candidate = np.tanh(np.concatenate((r_t * h_prev, x_t),
                                             axis=1) @ self.Wh + self.bh)

        # Compute next hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_candidate

        # Output using softmax
        y_linear = h_next @ self.Wy + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """Apply the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
