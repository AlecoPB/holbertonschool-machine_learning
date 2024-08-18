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
        """
        Kernel for the activation
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) +\
            np.sum(X2**2, axis=1) -\
            2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation
        of points in a Gaussian process
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y).flatten()

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        sigma_s = np.diag(cov_s)

        return mu_s, sigma_s

    def update(self, X_new, Y_new):
            """
            Update a Gaussian Process
            """
            # Update X and Y with the new data points
            self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
            self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))

            # Update the covariance matrix K with the new data point
            K_new = self.kernel(self.X, self.X)
            self.K = K_new
