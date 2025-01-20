#!/usr/bin/env python3
"""
This is some docuementation
"""
import tensorflow as tf
import numpy as np


def evaluate(X, Y, save_path):
    """evaluate
    
    X: data to evaluate
    Y: one_hot labels for X
    save_path: location of the model

    returns: the network's data
    """
    # Load the model
    model = tf.keras.models.load_model(save_path)

    # Evaluate the model
    loss, accuracy = model.evaluate(X, Y, verbose=0)

    # Get predictions
    predictions = model.predict(X)

    return predictions, accuracy, loss
