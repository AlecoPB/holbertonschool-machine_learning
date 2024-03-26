#!/usr/bin/env python3
"""_summary_
This is some documentation
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """_summary_

    Args:
        labels: Labels
        classes: Number of classes
    Returns:
        One hot encrypted matrix
    """
    return K.utils.to_categorical(labels, num_classes=classes)