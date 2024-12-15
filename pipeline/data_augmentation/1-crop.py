#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.
    """
    cropped_image = tf.image.random_crop(image, size)
    return cropped_image
