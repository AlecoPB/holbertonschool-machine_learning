#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """_summary_

    Args:
        network (_type_): _description_
        filename (_type_): _description_
        save_format (str, optional): _description_. Defaults to 'h5'.
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """_summary_

    Args:
        network (_type_): _description_
        filename (_type_): _description_
    """
    network.load_weights(filename)
