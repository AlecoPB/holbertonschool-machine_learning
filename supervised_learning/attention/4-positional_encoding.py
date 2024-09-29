#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate the positional encoding for a transformer.
    """
    pe = np.zeros((max_seq_len, dm))

    # Create an array of position indices
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Create an array of dimension indices
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Apply the sine function to even indices in the array; 2i
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply the cosine function to odd indices in the array; 2i+1
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
