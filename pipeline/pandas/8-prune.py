#!/usr/bin/env python3
"""
Removes NaN values
"""


def prune(df):
    """Removes NaN values

    Args:
        df (DataFrame)
    """
    return df.dropna()
