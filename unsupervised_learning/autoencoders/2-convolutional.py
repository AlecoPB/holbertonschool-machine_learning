#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Create a vanilla autoencoder
    """
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(latent_dims[0] * latent_dims[1] * latent_dims[2], activation='relu')(x)
    encoder = keras.Model(inputs, latent, name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims[0] * latent_dims[1] * latent_dims[2],))
    x = keras.layers.Dense(latent_dims[0] * latent_dims[1] * latent_dims[2], activation='relu')(latent_inputs)
    x = keras.layers.Reshape(latent_dims)(x)

    reversed_filters = filters[::-1]
    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(f, (3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(reversed_filters[-1], (3, 3), padding='valid', activation='relu')(x)
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid', padding='same')(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Autoencoder
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(auto_input, decoded, name="autoencoder")

    # Compile the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
