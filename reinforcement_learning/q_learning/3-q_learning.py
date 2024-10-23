#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100,
          alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    Training function
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range (max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            # Take action and observe new state and reward
            # Update to unpack 5 values and combine terminated and truncated to done
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated # done is True if either terminated or truncated is True


            if done and reward == 0:
                reward = -1

            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q, total_rewards
