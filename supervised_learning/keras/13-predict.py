#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network model.

    Args:
    - network: the network model to make the prediction with
    - data: the input data to make the prediction with
    - verbose: a boolean that determines if output should be
    printed during the prediction process

    Returns:
    - predictions: the predictions made by the model
    """
    predictions = network.predict(data, verbose=verbose)
    return predictions
