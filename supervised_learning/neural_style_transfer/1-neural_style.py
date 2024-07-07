#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class NST:
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.preprocess_image(style_image)
        self.content_image = self.preprocess_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    def preprocess_image(self, image):
        """
        Preprocesses the image for the model.
        """
        image = tf.convert_to_tensor(image)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.vgg19.preprocess_input(image)
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = tf.expand_dims(image, axis=0)
        return image

    def load_model(self):
        """
        Creates the model used to calculate cost using the VGG19 base model.
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Specify the layers to be used for style and content extraction
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.content_layer = 'block5_conv2'

        # Verify that specified layers exist in the VGG19 model
        available_layers = [layer.name for layer in vgg.layers]
        assert all(name in available_layers for name in self.style_layers), "One or more style layers are not in VGG19"
        assert self.content_layer in available_layers, "Content layer is not in VGG19"

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_output]

        model = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)
        return model