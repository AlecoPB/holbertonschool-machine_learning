#!/usr/bin/env python3
"""
This is some documentation
"""
import tensorflow.keras as keras


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder
    """
    # Encoder
    input_layer = keras.Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])
    encoder = keras.Model(inputs=input_layer, outputs=[z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_input = keras.Input(shape=(latent_dims,))
    x = latent_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_input, outputs=output_layer, name='decoder')

    # VAE Model
    auto_input = input_layer
    encoded = encoder(auto_input)[0]
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded, name='autoencoder')

    # Loss function
    reconstruction_loss = keras.losses.binary_crossentropy(auto_input, decoded)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)

    auto.compile(optimizer='adam')

    return encoder, decoder, auto
