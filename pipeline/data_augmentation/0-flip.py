#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.
    """
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image
