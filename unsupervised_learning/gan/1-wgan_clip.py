#!/usr/bin/env python3
"""
This is some doc
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class Simple_WGAN(keras.Model):
    
    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=5, learning_rate=.00005):
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.clip_value = 0.01  # Weight clipping range for WGAN
        
        # Optimizers
        self.generator.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        self.discriminator.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
    
    def train_step(self, data):
        # Training the discriminator (Critic)
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                
                discr_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
                
            gradients = tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
            
            # Clip discriminator weights
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))
        
        # Training the generator
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=False)
            
            gen_loss = -tf.reduce_mean(fake_output)
            
        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
