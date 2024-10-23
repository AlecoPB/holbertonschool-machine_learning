#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def q_init(env):
    """
    Initializes a q table with zeros
    """
    actions = env.action_space.n
    states = env.observation_space.n 
    return np.zeros((states, actions))
