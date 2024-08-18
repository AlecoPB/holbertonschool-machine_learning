#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Bayesian Optimization Class
    """
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound,
                               max_bound,
                               ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Bayesian Aquisition
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            improvement = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            improvement = mu - mu_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = improvement / sigma
            EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0  # When sigma is 0, EI should be 0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """_summary_

        Args:
            iterations (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Stop early if X_next has already been sampled
            if np.any(np.isclose(self.gp.X, X_next, atol=1e-8)):
                break

            # Sample the function at X_next
            Y_next = self.f(X_next)

            # Update the Gaussian Process with the new sample
            self.gp.update(X_next, Y_next)

        # Determine the optimal point
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
