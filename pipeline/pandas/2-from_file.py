#!/usr/bin/env python3
"""
Function to create a DataFrame from a file
"""
import pandas as pd


def from_file(filename, delimiter):
    """Creates a pd.DataFrame from a file

    Args:
        filename (string): file to load from
        delimiter : column separator
    """

    return pd.DataFrame(pd.read_csv(filename, delimiter=delimiter))
