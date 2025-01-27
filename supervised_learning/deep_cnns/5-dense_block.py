#!/usr/bin/env python3
"""
This is some documentation
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Constructs a dense block as outlined in
    'Densely Connected Convolutional Networks (2018)'
    """
    # Initialize he_normal with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    for layer_index in range(layers):
        # Batch normalization and ReLU activation prior to convolution
        batch_norm1 = K.layers.BatchNormalization()(X)
        relu_activation1 = K.layers.Activation(activation="relu")(batch_norm1)

        # 1x1 "bottleneck" convolution, with '4 x k' channels
        bottleneck_conv = K.layers.Conv2D(filters=4 * growth_rate,
                                           kernel_size=(1, 1),
                                           padding="same",
                                           kernel_initializer=initializer)(relu_activation1)

        # BatchNorm and ReLU before the 3x3 convolution
        batch_norm2 = K.layers.BatchNormalization()(bottleneck_conv)
        relu_activation2 = K.layers.Activation("relu")(batch_norm2)
        conv3x3 = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=(3, 3),
                                   padding="same",
                                   kernel_initializer=initializer)(relu_activation2)

        # Concatenate the inputs and new outputs along the channel axis
        X = K.layers.Concatenate()([X, conv3x3])

        # Update the filter count by the growth rate
        nb_filters += growth_rate

    return X, nb_filters
