#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
        labels (numpy.ndarray): One-hot encoded matrix of correct labels (shape: (m, classes)).
        logits (numpy.ndarray): One-hot encoded matrix of predicted labels (shape: (m, classes)).

    Returns:
        numpy.ndarray: Confusion matrix (shape: (classes, classes)).
    """
    m, classes = labels.shape
    confusion_matrix = np.zeros((classes, classes), dtype=np.float32)

    for i in range(m):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix
