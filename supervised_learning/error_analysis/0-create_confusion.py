#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    decoded_labels = np.zeros_like(labels, dtype=np.uint8)
    for i in range(labels.shape[0]):
        decoded_labels[i, np.argmax(labels[i])] = 1
    print(decoded_labels.shape, logits.shape)
    return decoded_labels.shape, logits.shape
