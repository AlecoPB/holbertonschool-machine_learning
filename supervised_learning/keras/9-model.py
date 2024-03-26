#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def save_model(network, filename):
    network.save(filename)

def load_model(filename):
    return K.models.load_model(filename)