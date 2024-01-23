#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""
This is some documentation
"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Args:
        x (_type_): _description_
        layer_sizes (list, optional): _description_. Defaults to [].
        activations (list, optional): _description_. Defaults to [].
    """
    layer_input = x
    for i in range(len(layer_sizes)):
        layer_output = create_layer(layer_input,
                                    layer_sizes[i], activations[i])
        layer_input = layer_output
    prediction = layer_output
    return prediction
