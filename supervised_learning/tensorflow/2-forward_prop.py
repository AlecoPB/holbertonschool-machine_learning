#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer
"""
This is some documentation
"""


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Args:
        x (_type_): _description_
        layer_sizes (list, optional): _description_. Defaults to [].
        activations (list, optional): _description_. Defaults to [].
    """
    layers = []
    for i in range(len(activations)):
        if i == 0:
            layers.append(create_layer(x, layer_sizes[i], activations[i]))
        else:
            layers.append(create_layer(layers[i - 1], layer_sizes[i], activations[i]))
    return layers
