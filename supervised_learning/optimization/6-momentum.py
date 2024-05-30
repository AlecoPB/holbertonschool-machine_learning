#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """_summary_

    Args:
        loss (_type_): _description_
        alpha (_type_): _description_
        beta1 (_type_): _description_
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    train_op = optimizer.minimize(loss)
    return train_op
