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
    for pos in range(max_seq_len):
        for i in range(dm):
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos / (10000 ** (i // 2 / dm)))
            else:
                pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) // 2 / dm)))
    return pe
