#!/usr/bin/env python3
"""
This is some documentation
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Constructs a projection block
    """
    F11, F3, F12 = filters

    # Initialize he_normal with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # First layer of the left branch (with stride s)
    layer1 = K.layers.Conv2D(filters=F11,
                             kernel_size=(1, 1),
                             strides=(s, s),
                             padding="same",
                             kernel_initializer=initializer)(A_prev)

    batch_norm1 = K.layers.BatchNormalization(axis=-1)(layer1)
    activation1 = K.layers.Activation(activation="relu")(batch_norm1)

    # Second layer of the left branch
    layer2 = K.layers.Conv2D(filters=F3,
                             kernel_size=(3, 3),
                             strides=(1, 1),
                             padding="same",
                             kernel_initializer=initializer)(activation1)
    batch_norm2 = K.layers.BatchNormalization(axis=-1)(layer2)
    activation2 = K.layers.Activation(activation="relu")(batch_norm2)

    # Final layer of the left branch
    layer3 = K.layers.Conv2D(filters=F12,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             padding="same",
                             kernel_initializer=initializer)(activation2)
    batch_norm3 = K.layers.BatchNormalization(axis=-1)(layer3)

    # Right branch: convolve input using F12 with stride s followed by BatchNorm
    conv_input = K.layers.Conv2D(filters=F12,
                                 kernel_size=(1, 1),
                                 strides=(s, s),
                                 padding="same",
                                 kernel_initializer=initializer)(A_prev)
    batch_norm_input = K.layers.BatchNormalization(axis=-1)(conv_input)

    # Combine outputs of the left and right branches
    combined = K.layers.Add()([batch_norm3, batch_norm_input])

    # Return the activated output of the combined layers using ReLU
    return K.layers.Activation(activation="relu")(combined)
