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
        self.model = self.load_model()
        self.gram_style_features, self.content_feature = self.generate_features()

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

    def load_model(self) -> tf.keras.Model:
        """
        Creates the model used to calculate cost using the VGG19 base model.

        Returns:
            tf.keras.Model: The Keras model used for Neural Style Transfer.
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)
        return model

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.

        Sets the instance attributes:
            gram_style_features - a list of gram matrices calculated from the style layer outputs of the style image
            content_feature - the content layer output of the content image
        """
        style_outputs = self.model(self.style_image)[:len(self.style_layers)]
        content_output = self.model(self.content_image)[len(self.style_layers):][0]

        gram_style_features = [self.gram_matrix(style_output) for style_output in style_outputs]
        content_feature = content_output

        return gram_style_features, content_feature

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Computes the Gram matrix of an input tensor.

        Args:
            input_tensor: A tensor of shape (1, height, width, channels).

        Returns:
            Gram matrix of the input tensor.
        """
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)
