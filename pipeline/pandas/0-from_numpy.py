#!/usr/bin/env python3
"""
Function to create a DataFrame from a np.array
"""
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray

    Args:
        array (np.ndarray): to transform into a DF
    """
    columns = [chr(i) for i in range(65, 91)]
    return pd.DataFrame(array, columns=columns)
