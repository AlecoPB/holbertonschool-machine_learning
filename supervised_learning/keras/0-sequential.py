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
    model = K.Sequential()
    
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            # First layer needs input_shape
            model.add(K.layers.Dense(nodes, activation=activation, kernel_regularizer=K.regularizers.l2(lambtha), input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(nodes, activation=activation, kernel_regularizer=K.regularizers.l2(lambtha)))
        
        # Add dropout layer if keep_prob is provided and it's not the last layer
        if keep_prob is not None and i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1-keep_prob))
    
    return model
