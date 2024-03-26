#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.Keras as K


def optimize_model(network, alpha, beta1, beta2):
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network
