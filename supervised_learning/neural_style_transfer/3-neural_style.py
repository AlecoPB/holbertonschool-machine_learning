#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for Neural Style Transfer.
    """
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the Neural Style Transfer class.
        """
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
        self.generate_features()

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

        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'

        outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]
        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

        return model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of the input layer.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        flattened = tf.reshape(input_layer, shape=(-1, input_layer.shape[-1]))
        gram = tf.matmul(flattened, flattened, transpose_a=True)
        gram /= tf.cast(tf.reduce_prod(input_layer.shape[1:]), dtype=tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        """
        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)

        style_layer_outputs = style_outputs[:-1]
        content_layer_output = content_outputs[-1]

        self.gram_style_features = [self.gram_matrix(output) for output in style_layer_outputs]
        self.content_feature = content_layer_output

    def style_loss(self, style_outputs, generated_outputs):
        """
        Calculates the style loss.
        """
        loss = 0
        for style_output, gen_output in zip(style_outputs, generated_outputs):
            gram_style = self.gram_matrix(style_output)
            gram_generated = self.gram_matrix(gen_output)
            loss += tf.reduce_mean(tf.square(gram_style - gram_generated))
        return loss
