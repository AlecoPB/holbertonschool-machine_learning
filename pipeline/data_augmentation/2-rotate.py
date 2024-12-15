#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.
    """
    rotated_image = tf.image.rot90(image)
    return rotated_image
