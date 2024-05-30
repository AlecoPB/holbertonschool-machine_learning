#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """_summary_

    Args:
        loss (_type_): _description_
        alpha (_type_): _description_
        beta1 (_type_): _description_
        beta2 (_type_): _description_
        epsilon (_type_): _description_
    """
    optimizer =\
        tf.train.AdamOptimizer(learning_rate=alpha,
                               beta1=beta1, beta2=beta2,
                               epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
