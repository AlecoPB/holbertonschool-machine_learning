#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def save_config(network, filename):
    """_summary_

    Args:
        network (_type_): _description_
        filename (_type_): _description_
    """
    config = network.get_config()
    with open(filename, 'w') as f:
        f.write(str(config))


def load_config(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_config(eval(config))
