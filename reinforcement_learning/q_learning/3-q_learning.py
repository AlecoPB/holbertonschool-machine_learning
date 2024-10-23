#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import gymnasium as gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    Training function
    """
    actions = env.action_space.n
    states = env.observation_space.n
    total_rewards = []

    for episode in range(episodes):
        # Random state
        state = np.random.randint(0, states)

        action = epsilon_greedy(Q, state, epsilon)

        next_state, reward = env.step(action)

        reward = -1 if reward != 1 else 1

        total_rewards.append[reward]

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

        epsilon -= epsilon_decay

    return Q, total_rewards
