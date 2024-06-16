#!/usr/bin/env python3
"""
DenseNet-121
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    'Densely Connected Convolutional Networks'.

    Parameters:
        growth_rate (int): The growth rate.
        compression (float): The compression factor.

    Returns:
        keras.Model: The Keras model of the DenseNet-121 architecture.
    """
    initializer = K.initializers.he_normal(seed=0)
    input_shape = (224, 224, 3)
    inputs = K.layers.Input(shape=input_shape)

    # Initial Convolution and Pooling
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer=initializer)(inputs)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, nb_filters=64,
                                growth_rate=growth_rate, layers_count=6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate,
                                layers_count=12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate,
                                layers_count=24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate,
                                layers_count=16)

    # Classification Layer
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.ReLU()(X)
    X = K.layers.GlobalAveragePooling2D()(X)
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=initializer)(X)

    # Create model
    model = K.models.Model(inputs=inputs, outputs=X)

    return model
