#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.
    """
    contrast_adjusted_image = tf.image.random_contrast(image, lower, upper)
    return contrast_adjusted_image
