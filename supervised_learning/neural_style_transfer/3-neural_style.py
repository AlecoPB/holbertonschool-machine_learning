#!/usr/bin/env python3
"""
This is some documenation
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    The NST class performs tasks for neural style transfer.

    Public Class Attributes:
    - style_layers: A list of layers to be used for style extraction,
    defaulting to ['block1_conv1', 'block2_conv1', 'block3_conv1',
    'block4_conv1', 'block5_conv1'].
    - content_layer: The layer to be used for content extraction,
    defaulting to 'block5_conv2'.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes an NST instance.

        Parameters:
        - style_image (numpy.ndarray): The image used as a style reference.
        - content_image (numpy.ndarray): The image used as a content reference
        - alpha (float): The weight for content cost. Default is 1e4.
        - beta (float): The weight for style cost. Default is 1.

        Raises:
        - TypeError: If style_image is not a numpy.ndarray with
            shape (h, w, 3), raises an error with the message "style_image
            must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If content_image is not a numpy.ndarray with
            shape (h, w, 3), raises an error with the message "content_image
            must be a numpy.ndarray with shape (h, w, 3)".
        - TypeError: If alpha is not a non-negative number, raises an error
            with the message "alpha must be a non-negative number".
        - TypeError: If beta is not a non-negative number, raises an error
            with the message "beta must be a non-negative number".

        Instance Attributes:
        - style_image: The preprocessed style image.
        - content_image: The preprocessed content image.
        - alpha: The weight for content cost.
        - beta: The weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Parameters:
        - image (numpy.ndarray): A numpy.ndarray of shape (h, w, 3) containing
        the image to be scaled.

        Raises:
        - TypeError: If image is not a numpy.ndarray with shape (h, w, 3),
          raises an error with the message "image must be a numpy.ndarray
          with shape (h, w, 3)".

        Returns:
        - tf.Tensor: The scaled image as a tf.Tensor with shape
          (1, h_new, w_new, 3), where max(h_new, w_new) == 512 and
          min(h_new, w_new) is scaled proportionately.
          The image is resized using bicubic interpolation, and its pixel
          values are rescaled from the range [0, 255] to [0, 1].
        """
        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        # Resize image (with bicubic interpolation)
        resized_image = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC)

        # Normalize pixel values to the range [0, 1]
        normalized_image = resized_image / 255

        # Clip values to ensure they are within [0, 1] range
        clipped_image = tf.clip_by_value(normalized_image, 0, 1)

        # Add batch dimension and return
        return tf.expand_dims(clipped_image, axis=0)

    def load_model(self):
        """
        Load the VGG19 model with AveragePooling2D instead of MaxPooling2D.
        """
        modelVGG19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        modelVGG19.trainable = False

        # Selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [modelVGG19.get_layer(name).output for name
                   in selected_layers]

        # Construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # replace MaxPooling layers by AveragePooling layers
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model('vgg_base.h5',
                                               custom_objects=custom_objects)

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
        Compute the Gram matrix for a specified tensor.

        Args:
        - input_layer: A tf.Tensor or tf.Variable with shape (1, h, w, c).

        Returns:
        - A tf.Tensor with shape (1, c, c) that represents the Gram matrix of
            input_layer.
        """
        # Check the rank and batch size of input_layer
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4
                or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")

        # compute gram matrix: (batch, height, width, channel)
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Normalize by the number of locations and return the gram tensor
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / nb_locations

    def generate_features(self):
        """
        Extract the features necessary for calculating the neural style cost.
        """
        # Ensure the images are properly preprocessed
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        style_outputs = self.model(preprocessed_style)[:-1]

        # Assign content_feature, no additional processing needed
        self.content_feature = self.model(preprocessed_content)[-1]

        # Calculate and assign Gram matrices for the style layer outputs
        self.gram_style_features = [self.gram_matrix(
            output) for output in style_outputs]
