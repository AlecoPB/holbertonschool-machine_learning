#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    lambtha = tf.get_variable("lambtha", shape=(), dtype=tf.float32)
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    l2_cost = cost + lambtha * tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    return l2_cost
