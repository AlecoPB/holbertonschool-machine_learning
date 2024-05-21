#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """_summary_

    Args:
        network: Model to be optimized
        alpha: Learning rate
        beta1: Adam first parameter
        beta2: Adam second parameter
    """
    optimizer = K.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1,
                                  beta_2=beta2)
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network
