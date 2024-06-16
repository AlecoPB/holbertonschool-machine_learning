#!/usr/bin/env python3
"""
Projection Block
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    'Deep Residual Learning for Image Recognition' (2015).

    Parameters:
        A_prev (tensor): The output from-> the previous layer.
        filters (tuple or list): A tuple or list
        containing F11, F3, F12, respectively:
                                 - F11: Number of
                                 filters in the first 1x1 convolution.
                                 - F3: Number of
                                 filters in the 3x3 convolution.
                                 - F12: Number of
                                 filters in the second 1x1 convolution as well as
                                        the 1x1 convolution in the shortcut connection.
        s (int): The stride of the first convolution
        in both the main path and the shortcut connection.

    Returns:
        tensor: The activated output of the projection block.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                      strides=(s), padding='same', kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                      strides=(1, 1), padding='same', kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                      strides=(1, 1), padding='same', kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                             strides=(s), padding='same', kernel_initializer=initializer)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Final step: Add shortcut value to the main path, and pass it through a ReLU activation
    X = K.layers.Add()([X, shortcut])
    X = K.layers.ReLU()(X)

    return X
