#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.
    """
    hue_adjusted_image = tf.image.adjust_hue(image, delta)
    return hue_adjusted_image
