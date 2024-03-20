#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost, lambtha, weights, L):
    """_summary_

    Args:
        cost (_type_): _description_
        lambtha (_type_): _description_
        weights (_type_): _description_
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
    l2_reg = 0
    for weight in weights:
        l2_reg += tf.nn.l2_loss(weight)
    cost_l2_reg = cost + lambtha * l2_reg
    return cost_l2_reg
