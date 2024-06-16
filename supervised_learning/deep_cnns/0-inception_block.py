#!/usr/bin/env python3

"""
This module implements an inception block for a neural network.
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as
    described in "Going Deeper with Convolutions (2014)".

    Args:
        A_prev: the output from the previous layer
        filters: a tuple or list containing
        F1, F3R, F3, F5R, F5, FPP, respectively

    Returns:
        The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(F1, (1, 1), activation='relu')(A_prev)

    conv3r = K.layers.Conv2D(F3R, (1, 1), activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3,
                            (3, 3),
                            padding='same',
                            activation='relu')(conv3r)

    conv5r = K.layers.Conv2D(F5R, (1, 1), activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5,
                            (5, 5),
                            padding='same',
                            activation='relu')(conv5r)

    pool = K.layers.MaxPooling2D((3, 3),
                                 strides=(1, 1),
                                 padding='same')(A_prev)
    pool_conv = K.layers.Conv2D(FPP, (1, 1), activation='relu')(pool)

    output = K.layers.Concatenate()([conv1, conv3, conv5, pool_conv])

    return output
