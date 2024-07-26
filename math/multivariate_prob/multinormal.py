#!/usr/bin/env python3
"""
This module contains the MultiNormal class that
represents a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Class constructor.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)


    def pdf(self, x):
        """
        Calculates the PDF at a data point.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** d * det_cov))
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered)

        return float(norm_const * np.exp(exponent))
