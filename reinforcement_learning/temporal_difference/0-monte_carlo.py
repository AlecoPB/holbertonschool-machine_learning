#!/usr/bin/env python3
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    # Initialize a dictionary to store returns for each state
    returns = {}
    
    for episode in range(episodes):
        state = env.reset()  # Reset the environment to start a new episode
        episode_data = []  # Store (state, reward) tuples for the episode
        
        for step in range(max_steps):
            action = policy(state)  # Get action from the policy
            next_state, reward, done, _ = env.step(action)  # Take action in the environment
            
            episode_data.append((state, reward))  # Store the state and reward
            state = next_state
            
            if done:
                break  # End the episode if done
        
        # Calculate the return G for each state in the episode
        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G  # Calculate the return
            
            if state not in returns:
                returns[state] = []
            returns[state].append(G)  # Store the return for the state
            
            # Update the value estimate for the state using the average of returns
            V[state] = np.mean(returns[state])  # Update to the average return
            
    return V