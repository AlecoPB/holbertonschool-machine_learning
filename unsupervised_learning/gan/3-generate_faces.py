#!/usr/bin/env python3
"""
This is some doc
"""
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Constructs a convolutional generator and discriminator using the functional API.

    Returns:
        - A generator model that converts a latent vector of shape (16) into an
        output of shape (16, 16, 1).
        - A discriminator model that processes an input of shape (16, 16, 1) to yield
        a single output (probability).
    """

    def build_gen_block(input_tensor, num_filters):
        """
        Constructs a block for the generator model.

        Args:
            input_tensor: The input tensor.
            num_filters: The number of filters for the Conv2D layer.

        Returns:
            - The output tensor after applying UpSampling2D, Conv2D,
            BatchNormalization, and Activation layers.
        """
        input_tensor = keras.layers.UpSampling2D()(input_tensor)
        input_tensor = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        input_tensor = keras.layers.BatchNormalization()(input_tensor)
        input_tensor = keras.layers.Activation('tanh')(input_tensor)
        return input_tensor

    def build_discr_block(input_tensor, num_filters):
        """
        Constructs a block for the discriminator model.

        Args:
            input_tensor: The input tensor.
            num_filters: The number of filters for the Conv2D layer.

        Returns:
            - The output tensor after applying Conv2D, MaxPooling2D,
            and Activation layers.
        """
        input_tensor = keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        input_tensor = keras.layers.MaxPooling2D()(input_tensor)
        input_tensor = keras.layers.Activation('tanh')(input_tensor)
        return input_tensor

    def get_generator():
        """
        Constructs the generator model using the functional API.

        Returns:
            - A generator model.
        """
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Apply 3 generator blocks with decreasing filter sizes
        x = build_gen_block(x, 64)
        x = build_gen_block(x, 16)
        x = build_gen_block(x, 1)

        # Create the generator model
        return keras.Model(inputs, x, name='generator')

    def get_discriminator():
        """
        Constructs the discriminator model using the functional API.

        Returns:
            - A discriminator model.
        """
        inputs = keras.Input(shape=(16, 16, 1))

        # Apply 4 discriminator blocks with increasing filter sizes
        x = build_discr_block(inputs, 32)
        x = build_discr_block(x, 64)
        x = build_discr_block(x, 128)
        x = build_discr_block(x, 256)

        # Flatten and apply a Dense layer with tanh activation for the final output
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        # Create the discriminator model
        return keras.Model(inputs, outputs, name='discriminator')

    return get_generator(), get_discriminator()