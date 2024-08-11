#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Baum Welch Algorithm
    """
    try:
        T = Observations.shape[0]  # Number of observations
        M = Transition.shape[0]    # Number of hidden states
        N = Emission.shape[1]      # Number of possible observations

        for _ in range(iterations):
            # E-Step: Calculate forward and backward probabilities
            # Forward part
            alpha = np.zeros((M, T))
            alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]

            for t in range(1, T):
                for j in range(M):
                    alpha[j, t] = (np.sum(alpha[:, t-1] * Transition[:, j])
                                   * Emission[j, Observations[t]])

            # Backward part
            beta = np.zeros((M, T))
            beta[:, T-1] = 1

            for t in range(T-2, -1, -1):
                for i in range(M):
                    beta[i, t] = (np.sum(Transition[i, :]
                                         * Emission[:, Observations[t+1]] * beta[:, t+1]))

            # Calculate gamma and xi
            gamma = np.zeros((M, T))
            xi = np.zeros((M, M, T-1))

            for t in range(T-1):
                denom = np.sum(alpha[:, t] * beta[:, t])
                for i in range(M):
                    gamma[i, t] = (alpha[i, t] * beta[i, t]) / denom
                    xi[i, :, t] = (alpha[i, t] * Transition[i, :]
                                   * Emission[:, Observations[t+1]] * beta[:, t+1]) / denom

            gamma[:, T-1] = alpha[:, T-1] * beta[:, T-1] / np.sum(alpha[:, T-1] * beta[:, T-1])

            # M-Step: Update the parameters
            # Update Initial probabilities
            Initial[:, 0] = gamma[:, 0]

            # Update Transition probabilities
            Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1).reshape(-1, 1)

            # Update Emission probabilities
            for k in range(N):
                mask = (Observations == k)
                Emission[:, k] = np.sum(gamma[:, mask], axis=1) / np.sum(gamma, axis=1)

        return Transition, Emission

    except Exception:
        return None, None
