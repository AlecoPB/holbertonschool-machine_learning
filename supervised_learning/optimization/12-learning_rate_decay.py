#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """_summary_

    Args:
        alpha (_type_): _description_
        decay_rate (_type_): _description_
        global_step (_type_): _description_
        decay_step (_type_): _description_
    """
    alpha_decay = tf.train.inverse_time_decay(learning_rate=alpha,
                                             global_step=global_step,
                                             decay_steps=decay_step,
                                             decay_rate=decay_rate)
    return alpha_decay
