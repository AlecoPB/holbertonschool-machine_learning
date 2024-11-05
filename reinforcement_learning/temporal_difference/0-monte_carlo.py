#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Monte Carlo policy evaluation to estimate the value function V under a given policy.
    
    Parameters:
        env: OpenAI Gym-like environment with `reset` and `step` methods.
        V: Dictionary representing the state value function.
        policy: Function that takes a state and returns an action.
        episodes: Number of episodes to sample.
        max_steps: Maximum steps per episode.
        alpha: Learning rate for incremental updates.
        gamma: Discount factor for future rewards.
        
    Returns:
        Updated value function V.
    """
    
    for episode in range(episodes):
        # Reset environment to get the initial state
        state = env.reset()
        episode_data = []  # To store (state, reward) tuples
        
        # Generate an episode
        for t in range(max_steps):
            action = policy(state)  # Get action from policy
            next_state, reward, done, _ = env.step(action)  # Take action in env
            
            episode_data.append((state, reward))  # Record (state, reward) pair
            state = next_state  # Update state
            
            if done:
                break  # End episode if 'done' signal is received
        
        # Process the episode to calculate returns and update V
        G = 0  # Initialize return (G_t) as 0
        visited_states = set()  # Track visited states in this episode for first-visit MC
        
        # Loop backwards through the episode data to calculate returns
        for state, reward in reversed(episode_data):
            G = reward + gamma * G  # Calculate return as G_t = R_(t+1) + gamma * G_(t+1)
            
            # First-visit MC: Update V only on first occurrence in the episode
            if state not in visited_states:
                visited_states.add(state)
                
                # Update the value function with incremental update
                V[state] = V.get(state, 0) + alpha * (G - V.get(state, 0))
    
    return V
