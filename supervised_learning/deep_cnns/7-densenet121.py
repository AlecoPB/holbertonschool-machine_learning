#!/usr/bin/env python3
"""
DenseNet-121
"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Constructs the DenseNet-121 architecture as detailed in
    'Densely Connected Convolutional Networks (2018)'.
    """
    # Initialize he_normal with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # Input tensor (assuming the given shape of data)
    input_tensor = K.Input(shape=(224, 224, 3))

    # First BN-ReLU-Conv, with double the growth rate for the initial filter count
    nb_filters = growth_rate * 2
    batch_norm = K.layers.BatchNormalization()(input_tensor)
    relu_activation = K.layers.Activation(activation="relu")(batch_norm)
    conv_layer = K.layers.Conv2D(filters=nb_filters,
                                  kernel_size=(7, 7),
                                  strides=(2, 2),
                                  padding="same",
                                  kernel_initializer=initializer)(relu_activation)

    max_pool_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                         strides=(2, 2),
                                         padding="same")(conv_layer)

    # Dense block 1, transition layer 1, and so forth until block 4
    # NOTE: nb_filters is updated (halved) by transition_layer
    block1, nb_filters = dense_block(max_pool_layer, nb_filters, growth_rate, 6)
    trans1, nb_filters = transition_layer(block1, nb_filters, compression)

    block2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(block2, nb_filters, compression)

    block3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(block3, nb_filters, compression)

    block4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    # Average Pooling (7x7 global)
    avg_pool_layer = K.layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1))(block4)

    # Fully Connected Layer, softmax
    output_layer = K.layers.Dense(units=1000, activation='softmax',
                                   kernel_initializer=initializer)(avg_pool_layer)

    return K.Model(inputs=input_tensor, outputs=output_layer)
