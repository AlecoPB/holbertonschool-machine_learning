#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """_summary_

    Args:
        alpha (_type_): _description_
        decay_rate (_type_): _description_
        global_step (_type_): _description_
        decay_step (_type_): _description_

    Returns:
        _type_: _description_
    """
    decayed_alpha = alpha / (1 + decay_rate *
                             (global_step / decay_step))
    return decayed_alpha
