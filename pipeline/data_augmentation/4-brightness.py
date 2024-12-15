#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.
    """
    brightness_adjusted_image = tf.image.random_brightness(image, max_delta)
    return brightness_adjusted_image
