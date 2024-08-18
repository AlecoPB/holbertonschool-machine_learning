#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


class GaussianProcess:
    """
    Represent a noiseless 1D Gaussin process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) +\
            np.sum(X2**2, axis=1) -\
            2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
