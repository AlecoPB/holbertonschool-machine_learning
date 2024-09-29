#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Initialize the positional encoding matrix
    """
    pos_enc = np.zeros((max_seq_len, dm))

    # Calculate positional encoding values for each position
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            angle = pos / np.power(10000, (2 * i) / dm)
            pos_enc[pos, i] = np.sin(angle)
            if i + 1 < dm:
                pos_enc[pos, i + 1] = np.cos(angle)

    return pos_enc
