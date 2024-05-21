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
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model
