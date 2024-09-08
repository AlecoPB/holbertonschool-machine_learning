#!/usr/bin/env python3
"""Module containing the LSTMCell class."""

import numpy as np


class LSTMCell:
    """Represents a Long Short-Term Memory (LSTM) cell."""

    def __init__(self, i, h, o):
        """Initialize the LSTMCell.

        Parameters:
        i (int): Dimensionality of the data.
        h (int): Dimensionality of the hidden state.
        o (int): Dimensionality of the outputs.
        """
        # Forget gate weights and biases
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        # Update gate weights and biases
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        # Intermediate cell state weights and biases
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        # Output gate weights and biases
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Parameters:
        h_prev (ndarray): Previous hidden state, shape (m, h).
        c_prev (ndarray): Previous cell state, shape (m, h).
        x_t (ndarray): Data input for the cell, shape (m, i).

        Returns:
        h_next (ndarray): Next hidden state, shape (m, h).
        c_next (ndarray): Next cell state, shape (m, h).
        y (ndarray): Output of the cell, shape (m, o).
        """
        m, h = h_prev.shape
        x_h_concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(x_h_concat @ self.Wf + self.bf)

        # Update gate
        u_t = self.sigmoid(x_h_concat @ self.Wu + self.bu)

        # Candidate cell state
        c_tilde = np.tanh(x_h_concat @ self.Wc + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = self.sigmoid(x_h_concat @ self.Wo + self.bo)

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        # Output using softmax
        y_linear = h_next @ self.Wy + self.by
        y_exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """Apply the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
