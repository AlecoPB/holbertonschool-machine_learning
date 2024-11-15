#!/usr/bin/env python3
"""
This is some documenation
"""
import numpy as np
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
    scores = []

    # Determine dimensions for the weight matrix
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Generate a random weight matrix
    weight = np.random.randn(state_size, num_actions)

    for episode in range(nb_episodes):
        
        state = env.reset()[0]
        rewards, gradients = [], []
        ep_rewards = []

        # Initialize and reset G
        G = 0

        done = False

        while not done:
            # Decide action and gradient using policy_gradient
            action, gradient = policy_gradient(state, weight)

            # Take step and record state and rewards
            new_state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            gradients.append(gradient)

            # Update state
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
            weight += alpha * gradient * reward

    return scores
