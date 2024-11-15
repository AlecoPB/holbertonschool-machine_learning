#!/usr/bin/env python3
"""
This is some documenation
"""
import numpy as np
import random
policy_gradient = __import__('Policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Full training

    Args:
        env: initial environment
        nb_episodes (int): number of episodes used for training
        alpha (float, optional): learning rate. Defaults to 0.000045.
        gamma (float, optional): discount factor. Defaults to 0.98.

    Returns:
        sum of all rewards during one episode loop
    """
    scores, l_G = [], []
    # Determine the max episode steps
    max_steps = env.spec.max_episode_steps

    # Determine dimensions for the weight matrix
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Generate a random weight matrix
    weight = np.random.randn(num_actions, state_size)

    for episode in range(nb_episodes):
        state = env.reset()[0]
        rewards, states, gradients = [], [state], []
        
        # Initialize and reset G
        G = 0

        for _ in range(max_steps):
            # Decide action and gradient using policy_gradient
            action, gradient = policy_gradient(state, weight)

            # Take step and record state and rewards
            new_state, reward, done, truncated, _ = env.step(action)
            states.append(new_state)
            rewards.append(reward)
            gradients.append(gradient)

            # Update state
            state = new_state

        # Compute total reward for the episode
        scores.append(sum(rewards))

        # Calculate Discounted Reward (G)
        for r in reversed(rewards):
            G = r + gamma * G
            # print(f'G is currently: {}', G)
            l_G.insert(0, G)

        for gradient, G in zip(gradients, l_G):
            weight += alpha * gradient * G

    return scores