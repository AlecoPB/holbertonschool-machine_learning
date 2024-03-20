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
    reg_cost = 0
    for i in range(1, L + 1):
        reg_cost += tf.nn.l2_loss(weights['W' + str(i)])
    cost_l2 = cost + lambtha * reg_cost
    return cost_l2
