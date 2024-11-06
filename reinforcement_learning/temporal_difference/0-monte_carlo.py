#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Monte Carlo policy evaluation to estimate the value function V under a given policy.

    Parameters:
        env: Environment with `reset` and `step` methods.
        V: NumPy array representing the state value function.
        policy: Function that takes a state and returns an action.
        episodes: Number of episodes to sample.
        max_steps: Maximum steps per episode.
        alpha: Learning rate for incremental updates.
        gamma: Discount factor for future rewards.

    Returns:
        Updated value function V.
    """

    for episode in range(episodes):
        state = env.reset()
        # Handle environments returning a tuple (state, info)
        if isinstance(state, tuple):
            state = state[0]

        episode_data = []  # Store (state, reward) tuples

        # Generate an episode
        for t in range(max_steps):
            action = policy(state)  # Get action from policy
            step_output = env.step(action)

            # Unpack environment output to handle single or multiple returns
            next_state = step_output[0]
            reward = step_output[1]
            done = step_output[2] if len(step_output) > 2 else False

            episode_data.append((state, reward))  # Record (state, reward) pair
            state = next_state  # Update state

            if done:
                break  # End episode if 'done' signal is received

        # Process the episode to calculate returns and update V
        G = 0  # Initialize return (G_t) as 0
        visited_states = set()  # Track visited states for first-visit MC

        # Loop backwards through episode data
        for state, reward in reversed(episode_data):
            G = reward + gamma * G  # Compute return G_t

            # First-visit MC: Update V on first occurrence only
            if state not in visited_states:
                visited_states.add(state)

                # Update value function V
                V[state] = V[state] + alpha * (G - V[state])

    return V