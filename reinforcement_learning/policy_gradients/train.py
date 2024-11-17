#!/usr/bin/env python3
"""
Training loop for MC policy gradient
"""
import numpy as np
import gymnasium as gym
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Full training

    Args:
        env: initial environment
        nb_episodes (int): number of episodes used for training
        alpha (float, optional): learning rate. Defaults to 0.000045.
        gamma (float, optional): discount factor. Defaults to 0.98.

    Returns:
        sum of all rewards during one episode loop
    """
    # Initialize weights
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        # Display every 1000 episodes if show_result is true
        if show_result and i % 1000 == 0:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env = gym.make('CartPole-v1', render_mode=None)

        # Reset the environment and get initial state
        state = env.reset()[0]
        gradients, rewards = [], []

        # We set done to False so it won't break the loop instantly
        done = False

        while not done:
            # Decide action and gradient using policy_gradient
            action, gradient = policy_gradient(state, weights)

            # Take step and record state and rewards
            new_state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            gradients.append(gradient)

            state = new_state
            done = done or truncated

        # Compute total reward for the episode
        scores.append(sum(rewards))

        # Print current episode and score
        print(f'Episode: {episode} Score: {sum(rewards)}')

        # Update weights using the gradients and cumulative discounted rewards
        for i, gradient in enumerate(gradients):
            # Calculate cumulative discounted rewards
            reward = sum([R * gamma ** R for R in rewards[i:]])

            # Update the weights
            weights += alpha * gradient * reward

    return scores
