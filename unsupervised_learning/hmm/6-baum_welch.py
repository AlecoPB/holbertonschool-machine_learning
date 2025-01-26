#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def forward_algorithm(observations, emission_probs,
                      transition_probs, initial_probs):
    """
    Executes the forward algorithm for a hidden Markov model.
    """
    if not isinstance(observations, np.ndarray) or observations.ndim != 1:
        return None, None
    if not isinstance(emission_probs, np.ndarray) or emission_probs.ndim != 2:
        return None, None
    if not isinstance(transition_probs,
                      np.ndarray) or transition_probs.ndim != 2:
        return None, None
    if not isinstance(initial_probs, np.ndarray) or initial_probs.ndim != 2:
        return None, None

    N, M = emission_probs.shape
    T = observations.shape[0]

    if transition_probs.shape != (N, N):
        return None, None
    if initial_probs.shape != (N, 1):
        return None, None

    # Initialize the forward probability matrix
    forward_probs = np.zeros((N, T))

    # Fill the first column of forward_probs
    forward_probs[:, 0] = (initial_probs[:, 0]
                           * emission_probs[:, observations[0]])

    # Recursion to compute forward probabilities
    for t in range(1, T):
        # Calculate the next column using the previous column
        forward_probs[:, t] = (forward_probs[:, t-1] @ transition_probs
                               * emission_probs[:, observations[t]])

    # The probability of the observation sequence is the
    # sum of the last column in forward_probs
    likelihood = np.sum(forward_probs[:, -1])

    return likelihood, forward_probs


def backward_algorithm(observations, emission_probs,
                       transition_probs, initial_probs):
    """
    Executes the backward algorithm for a hidden Markov model.
    """
    if (not isinstance(observations, np.ndarray) or observations.ndim != 1 or
            not isinstance(emission_probs, np.ndarray)
            or emission_probs.ndim != 2 or
            not isinstance(transition_probs, np.ndarray)
            or transition_probs.ndim != 2 or
            not isinstance(initial_probs, np.ndarray)
            or initial_probs.ndim != 2):
        return None, None

    N, M = emission_probs.shape
    T = observations.shape[0]

    if transition_probs.shape != (N, N):
        return None, None
    if initial_probs.shape != (N, 1):
        return None, None

    # Initialize the backward probability matrix
    backward_probs = np.zeros((N, T))

    # Set the last column of backward_probs to 1
    backward_probs[:, T - 1] = 1

    # Recursion: Fill backward_probs from time T-2 down to time 0
    for t in range(T - 2, -1, -1):
        # Calculate backward probabilities for states at time t
        backward_probs[:, t] = np.sum(
            transition_probs * emission_probs[:, observations[t + 1]]
            * backward_probs[:, t + 1], axis=1)

    # Calculate the likelihood of the observations given the model
    likelihood = np.sum(initial_probs[:, 0]
                        * emission_probs[:, observations[0]]
                        * backward_probs[:, 0])

    return likelihood, backward_probs


def baum_welch(observations, transition_probs, emission_probs,
               initial_probs, num_iterations=1000):
    """
    Executes the Baum-Welch algorithm for a hidden Markov model.
    """
    if (not isinstance(observations, np.ndarray) or observations.ndim != 1 or
            not isinstance(emission_probs, np.ndarray)
            or emission_probs.ndim != 2 or
            not isinstance(transition_probs, np.ndarray)
            or transition_probs.ndim != 2 or
            not isinstance(initial_probs, np.ndarray)
            or initial_probs.ndim != 2):
        return None, None

    N = transition_probs.shape[0]
    M = emission_probs.shape[1]
    T = observations.shape[0]

    for _ in range(num_iterations):
        # Perform forward and backward passes
        likelihood_f, forward_probs = forward_algorithm(observations,
                                                        emission_probs,
                                                        transition_probs,
                                                        initial_probs)
        likelihood_b, backward_probs = backward_algorithm(observations,
                                                          emission_probs,
                                                          transition_probs,
                                                          initial_probs)

        # Initialize variables for xi and gamma
        xi = np.zeros((N, N, T-1))
        gamma = np.zeros((N, T))

        for t in range(T-1):
            # Compute xi for all states
            xi[:, :, t] = ((forward_probs[:, t, np.newaxis]
                            * transition_probs
                            * emission_probs[:, observations[t+1]]
                            * backward_probs[:, t+1]) / likelihood_f)

        gamma = np.sum(xi, axis=1)

        # Include the final gamma element for the new backward probabilities
        prod = ((forward_probs[:, T-1]
                 * backward_probs[:, T-1]).reshape((-1, 1)))
        gamma = np.hstack((gamma, prod / np.sum(prod)))

        # Update the Transition matrix
        transition_probs = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))

        # Update the Emission matrix
        for k in range(M):
            emission_probs[:, k] = np.sum(gamma[:, observations == k], axis=1)

        emission_probs /= np.sum(gamma, axis=1).reshape(-1, 1)

    return transition_probs, emission_probs
