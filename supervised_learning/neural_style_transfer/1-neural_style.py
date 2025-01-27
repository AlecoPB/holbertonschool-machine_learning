#!/usr/bin/env python3
"""
Neural Style Transfer Module
"""

import numpy as np
import tensorflow as tf


class NeuralStyleTransfer:
    """
    Implements functionality for Neural Style Transfer (NST).

    Class Attributes:
    - style_layers: Layers used for extracting style features.
    - content_layer: Layer used for extracting content features.
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 
        'block3_conv1', 'block4_conv1', 
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NeuralStyleTransfer object.

        Parameters:
        - style_image (numpy.ndarray): Style reference image.
        - content_image (numpy.ndarray): Content reference image.
        - alpha (float): Weight for content cost.
        - beta (float): Weight for style cost.

        Raises:
        - TypeError: For invalid types or shapes of inputs.
        """
        self._validate_image(style_image, "style_image")
        self._validate_image(content_image, "content_image")
        self._validate_weight(alpha, "alpha")
        self._validate_weight(beta, "beta")

        self.style_image = self._process_image(style_image)
        self.content_image = self._process_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self._build_model()

    @staticmethod
    def _validate_image(image, name):
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError(f"{name} must be a numpy.ndarray with shape (h, w, 3)")

    @staticmethod
    def _validate_weight(value, name):
        if not isinstance(value, (float, int)) or value < 0:
            raise TypeError(f"{name} must be a non-negative number")

    @staticmethod
    def _process_image(image):
        """
        Rescales and preprocesses an image.

        Parameters:
        - image (numpy.ndarray): Image to process.

        Raises:
        - TypeError: If the image is not a numpy.ndarray with the correct shape.

        Returns:
        - tf.Tensor: Preprocessed image tensor with shape (1, h_new, w_new, 3).
        """
        NeuralStyleTransfer._validate_image(image, "image")

        h, w = image.shape[:2]
        scale = 512 / max(h, w)
        new_dims = (int(w * scale), int(h * scale))

        resized = tf.image.resize(image, new_dims, method='bicubic') / 255.0
        clipped = tf.clip_by_value(resized, 0, 1)
        return tf.expand_dims(clipped, axis=0)

    def _build_model(self):
        """
        Builds a modified VGG19 model for NST.
        Replaces MaxPooling layers with AveragePooling.
        """
        base_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        base_model.trainable = False

        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        style_outputs = [base_model.get_layer(name).output for name in self.style_layers]
        content_output = base_model.get_layer(self.content_layer).output

        self.model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=style_outputs + [content_output]
        )
