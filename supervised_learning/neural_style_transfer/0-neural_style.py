#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


class NST:
    """
    A class to perform Neural Style Transfer.

    Attributes:
        style_layers (list): List of layers to be used for style extraction.
        content_layer (str): Layer to be used for content extraction.
    """

    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image: np.ndarray, content_image: np.ndarray, alpha: float = 1e4, beta: float = 1):
        """
        Initializes the NST class with style and content images and their weights.

        Args:
            style_image (np.ndarray): The style reference image.
            content_image (np.ndarray): The content reference image.
            alpha (float): The weight for content cost.
            beta (float): The weight for style cost.
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image: np.ndarray) -> tf.Tensor:
        """
        Rescales an image such that its pixels values are between 0 and 1 and its largest side is 512 pixels.

        Args:
            image (np.ndarray): The image to be scaled.

        Returns:
            tf.Tensor: The scaled image.
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))

        image = tf.image.resize(image, (new_h, new_w), method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.expand_dims(image, axis=0)

        return image
