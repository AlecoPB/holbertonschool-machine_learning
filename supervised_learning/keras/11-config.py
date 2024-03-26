#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def save_config(network, filename):
    # Serialize model to JSON
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)

def load_config(filename):
    # Load json and create model
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    return loaded_model