#!/usr/bin/env python3
"""
Transition Layer
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    'Densely Connected Convolutional Networks'.
    
    Parameters:
        X (tensor): The output from the previous layer.
        nb_filters (int): The number of filters in X.
        compression (float): The compression factor for
        the transition layer.
    
    Returns:
        tensor: The output of the transition layer.
        int: The number of filters within the output.
    """
    initializer = K.initializers.he_normal(seed=0)
    
    # Compute the number of filters after compression
    nb_filters = int(nb_filters * compression)

    # Batch normalization
    X = K.layers.BatchNormalization(axis=3)(X)

    # ReLU activation
    X = K.layers.ReLU()(X)

    # 1x1 Convolution
    X = K.layers.Conv2D(filters=nb_filters, kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)

    # Average Pooling
    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding='same')(X)

    return X, nb_filters
