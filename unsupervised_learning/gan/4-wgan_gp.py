#!/usr/bin/env python3
"""
This is some doc
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
    """
    Implementation of a Wasserstein GAN (WGAN) with Gradient Penalty.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005, lambda_gp=10):
        """
        Initializes WGAN-GP with specified components and hyperparameters.
        """
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.lambda_gp = lambda_gp
        self.beta_1, self.beta_2 = 0.3, 0.9

        self.dims = tf.shape(real_examples)
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, dtype='int32')
        self.scal_shape = [batch_size] + [1] * (self.len_dims - 1)
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer =\
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        self.discriminator.loss = lambda real, fake: (tf.reduce_mean(fake)
                                                      - tf.reduce_mean(real))
        self.discriminator.optimizer =\
            keras.optimizers.Adam(learning_rate=self.learning_rate,
                                  beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Produces a batch of generated samples.
        """
        size = size or self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples from the dataset.
        """
        size = size or self.batch_size
        indices = tf.random.\
            shuffle(tf.range
                    (tf.shape
                     (self.real_examples)[0]))[:size]
        return tf.gather(self.real_examples, indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Creates interpolated samples between real and fake examples.
        """
        alpha = tf.random.uniform(self.scal_shape)
        return alpha * real_sample + (1 - alpha) * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Computes gradient penalty for the interpolated samples.
        """
        with tf.GradientTape() as tape:
            tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        gradients = tape.gradient(pred, [interpolated_sample])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                          axis=self.axis))
        return tf.reduce_mean((grad_norm - 1.0) ** 2)

    def train_step(self, _):
        """
        Executes a single training step for generator and discriminator.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_batch = self.get_real_sample()
                fake_batch = self.get_fake_sample(training=True)
                interpolated_batch = self.\
                    get_interpolated_sample(real_batch,
                                            fake_batch)

                real_pred = self.discriminator(real_batch, training=True)
                fake_pred = self.discriminator(fake_batch, training=True)
                disc_loss = self.discriminator.loss(real_pred, fake_pred)

                gp = self.gradient_penalty(interpolated_batch)
                total_disc_loss = disc_loss + self.lambda_gp * gp

            disc_gradients = disc_tape.\
                gradient(total_disc_loss,
                         self.discriminator.trainable_variables)
            self.discriminator.\
                optimizer.apply_gradients(zip(disc_gradients,
                                              self.discriminator.
                                              trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_batch = self.get_fake_sample(training=True)
            fake_pred = self.discriminator(fake_batch, training=False)
            gen_loss = self.generator.loss(fake_pred)

        gen_gradients = gen_tape.gradient(gen_loss,
                                          self.generator.trainable_variables)
        self.generator.optimizer.\
            apply_gradients(zip(gen_gradients,
                                self.generator.trainable_variables))

        return {"discr_loss": disc_loss, "gen_loss": gen_loss, "gp": gp}

    def replace_weights(self, gen_h5, disc_h5):
        """
        Loads weights for the generator and discriminator from specified files.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
