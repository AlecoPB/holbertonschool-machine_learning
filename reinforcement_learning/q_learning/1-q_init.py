#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
# load_frozen_lake =  '0-load_env'.load_frozen_lake


def q_init(env):
    """
    Initializes a q table with zeros
    """
    states, actions = env.observation_space.n, env.action_space.n
    return np.zeros(states, actions)

