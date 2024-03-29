#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """_summary_

    Args:
        nx (_type_): _description_
        layers (_type_): _description_
        activations (_type_): _description_
        lambtha (_type_): _description_
        keep_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    regularizer = K.regularizers.l2(lambtha)
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(nodes,
                           activation=activation,
                           kernel_regularizer=regularizer)(x)

        if keep_prob is not None and i < len(layers) - 1:
            x = K.layers.Dropout(rate=1-keep_prob)(x)

    outputs = x
    model = K.Model(inputs=inputs, outputs=outputs)

    return model
