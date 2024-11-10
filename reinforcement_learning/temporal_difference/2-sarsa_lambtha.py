#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform the SARSA(λ) algorithm to update the value function V.

    Parameters:
    - env: environment instance
    - Q (nump.ndarray) = Q table of shape (s, a)
    - lambtha (float): eligibility trace factor
    - episodes (int): number of episodes to train over
    - max_steps (int): max steps per episode
    - alpha (float): learning rate
    - gamma (float): discount rate
    - epsilon : initial threshold for epsilon greedy
    - min_epsilon : minimum value for epsilon
    - epsilon_decay : decay rate for epsilon between episodes

    Returns:
    - Q : updated Q table
    """
    for episode in range(episodes):
        state, _ = env.reset()

        # Update epsilon until we hit the minimum value
        epsilon = max(epsilon - epsilon_decay, min_epsilon)

