#!/usr/bin/env python3
"""
Bayesian Optimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    This class implements the Bayesian Optimization technique.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Initializes the Bayesian Optimization process.

        Parameters:
        - f: The black-box function to optimize.
        - X_init: numpy.ndarray of shape (t, 1), initial sampled inputs.
        - Y_init: numpy.ndarray of shape (t, 1), outputs of the black-box
        function corresponding to each input.
        - bounds: tuple of (min, max), defining the search space for the
        optimal
        point.
        - ac_samples: int, the number of samples for the acquisition function.
        - l: float, the length scale parameter for the kernel.
        - sigma_f: float, the standard deviation of the output.
        - xsi: float, the exploration-exploitation trade-off factor.
        - minimize: bool, indicates whether to minimize or maximize the
        function.
        """
        self.f = f
        self.gp = GP(X_init=X_init, Y_init=Y_init, l=l, sigma_f=sigma_f)
        # Generate an array of evenly spaced sampling values within the bounds
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Determine the next best sample location using the Expected Improvement
        (EI) acquisition function.

        This function balances exploration and exploitation by evaluating the
        expected improvement over the best current observation, depending on
        whether the goal is minimization or maximization.

        Returns:
        --------
        - X_next : numpy.ndarray of shape (1,), the next best sample point to
        evaluate the black-box function.

        - EI : numpy.ndarray of shape (ac_samples,) the expected improvement
        values for each point in the acquisition sample points (X_s).
        """
        # Predict the mean and variance for each point in X_s
        mu_s, sigma_s = self.gp.predict(self.X_s)

        # Determine the best observed value based on min-maxxing
        if self.minimize:
            Y_best = np.min(self.gp.Y)
            improvement = Y_best - mu_s - self.xsi
        else:
            Y_best = np.max(self.gp.Y)
            improvement = mu_s - Y_best - self.xsi

        # Calculate Z and Expected Improvement
        Z = improvement / sigma_s
        EI = improvement * norm.cdf(Z) + sigma_s * norm.pdf(Z)

        # Identify the next best point to sample
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimize the black-box function using Bayesian Optimization.

        This method iteratively proposes the next point to sample by
        maximizing the Expected Improvement (EI) acquisition function,
        updates the Gaussian Process model with the new sample, and continues
        for a specified number of iterations or until the next proposed point
        has already been sampled.

        Parameters:
        -----------
        - iterations : int, optional (default=100) the maximum number of
        iterations to perform during the optimization process.

        Returns:
        --------
        - X_opt : numpy.ndarray of shape `(1,)` the optimal point found during
        the optimization process.
        - Y_opt : numpy.ndarray of shape `(1,)` the function value at the
        optimal point.
        """
        for _ in range(iterations):
            # Identify the next best point to sample
            X_next, _ = self.acquisition()

            # Exit early if the next point has already been sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            # Evaluate the function at the proposed point
            Y_next = self.f(X_next)

            # Update the Gaussian Process with the new sample
            self.gp.update(X_next, Y_next)

        # Determine the optimal point and its corresponding function value
        if self.minimize:
            optimal_idx = np.argmin(self.gp.Y)
        else:
            optimal_idx = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1, :]
        X_opt = self.gp.X[optimal_idx]
        Y_opt = self.gp.Y[optimal_idx]

        return X_opt, Y_opt
