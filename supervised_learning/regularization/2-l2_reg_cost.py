#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """_summary_

    Args:
        cost (_type_): _description_
        lambtha (_type_): _description_
        weights (_type_): _description_
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
        # f_cost.append = tf.losses.get_regularization_loss(cost)
    return cost + tf.losses.get_regularization_loss()
