#!/usr/bin/env python3
"""_summary_

Returns:
    _type_: _description_
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Args:
        nx (int): number of feature columns in our data
        classes (int): number of classes in our classifier

    Returns:
        _type_: placeholders 
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')
    return x, y
