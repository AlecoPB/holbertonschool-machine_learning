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
    actions = env.action_space.n
    states = env.observation_space.n 
    return np.zeros(actions, states)


