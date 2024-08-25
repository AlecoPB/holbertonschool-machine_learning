#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Create a vanilla autoencoder
    """
    input_layer = keras.Input(shape=input_dims)
    x = input_layer
    for num_filters in filters:
        x = keras.layers.Conv2D(num_filters, (3, 3),
                                activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    latent_layer = x
    encoder = keras.Model(inputs=input_layer, outputs=latent_layer)

    # Decoder
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    for i, num_filters in enumerate(reversed(filters)):
        if i < len(filters) - 1:
            x = keras.layers.Conv2D(num_filters, (3, 3),
                                    activation='relu',
                                    padding='same')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
        else:
            x = keras.layers.Conv2D(num_filters, (3, 3),
                                    activation='relu',
                                    padding='valid')(x)
            x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[2], (3, 3),
                            activation='sigmoid',
                            padding='same')(x)
    output_layer = x
    decoder = keras.Model(inputs=latent_input, outputs=output_layer)

    auto_input = input_layer
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
