#!/usr/bin/env python3
"""
This is some documenation
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Full training

    Args:
        env: initial environment
        nb_episodes (int): number of episodes used for training
        alpha (float, optional): learning rate. Defaults to 0.000045.
        gamma (float, optional): discount factor. Defaults to 0.98.

    Returns:
        _type_: _description_
    """
    
    return 0