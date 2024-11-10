#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Selects action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])  # Random action
    else:
        return np.argmax(Q[state])  # Greedy action

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Perform SARSA(λ) algorithm with epsilon-greedy exploration.
    
    Parameters:
    - env: environment instance
    - Q (numpy.ndarray): array of shape (s, a) with state-action values
    - lambtha (float): eligibility trace factor
    - episodes (int): number of episodes to train over
    - max_steps (int): max steps per episode
    - alpha (float): learning rate
    - gamma (float): discount rate
    - epsilon (float): initial epsilon value for epsilon-greedy
    - min_epsilon (float): minimum epsilon value
    - epsilon_decay (float): rate of epsilon decay per episode
    
    Returns:
    - Q (numpy.ndarray): updated Q table
    """
    
    for episode in range(episodes):
        # Initialize eligibility traces for each state-action pair to 0
        eligibility_trace = np.zeros_like(Q)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        
        # Start a new episode
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        
        for step in range(max_steps):
            # Take the action in the environment
            (next_state, _), reward, done, _ = env.step(action)
            
            # Choose next action using epsilon-greedy policy
            next_action = epsilon_greedy(Q, next_state, epsilon)
            
            # Calculate the TD error (delta)
            delta = reward + gamma * Q[next_state, next_action] * (1 - int(done)) - Q[state, action]
            
            # Update eligibility trace for the current state-action pair
            eligibility_trace[state, action] += 1
            
            # Update Q-values for all state-action pairs
            Q += alpha * delta * eligibility_trace
            
            # Decay eligibility trace for all state-action pairs
            eligibility_trace *= gamma * lambtha
            
            # Move to the next state and action
            state, action = next_state, next_action
            
            # If the episode is finished, break out of the loop
            if done:
                break

    return Q
