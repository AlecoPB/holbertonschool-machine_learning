#!/usr/bin/env python3
"""
Dense Block
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers_count):
    """
    Builds a dense block as described in 'Densely Connected Convolutional Networks'.
    
    Parameters:
        X (tensor): The output from the previous layer.
        nb_filters (int): The number of filters in X.
        growth_rate (int): The growth rate for the dense block.
        K.layers_count (int): The number of K.layers in the dense block.
    
    Returns:
        tensor: The concatenated output of each layer within the Dense Block.
        int: The number of filters within the concatenated outputs.
    """
    initializer = K.initializers.he_normal(seed=0)

    for _ in range(layers_count):
        # Bottleneck layer
        bottleneck = K.layers.BatchNormalization(axis=3)(X)
        bottleneck = K.layers.ReLU()(bottleneck)
        bottleneck = K.layers.Conv2D(4 * growth_rate, (1, 1),
                                     padding='same',
                                     kernel_initializer=initializer)(bottleneck)

        # 3x3 Convolution layer
        conv = K.layers.BatchNormalization(axis=3)(bottleneck)
        conv = K.layers.ReLU()(conv)
        conv = K.layers.Conv2D(growth_rate, (3, 3),
                               padding='same',
                               kernel_initializer=initializer)(conv)

        # Concatenate input with output of this layer
        X = K.layers.Concatenate(axis=3)([X, conv])
        nb_filters += growth_rate

    return X, nb_filters
