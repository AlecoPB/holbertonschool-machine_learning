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
    # weights = [v for v in tf.trainable_variables()]
    # l2_reg_cost = cost
    # for i in range(0, 3):
    #     print(cost(i))
    # for w in weights:
    #     l2_reg_cost += tf.nn.l2_loss(w)
    dense = tf.keras.layers.Dense(3, kernel_regularizer='l2')
    cost *= dense
    return cost
