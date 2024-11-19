#!/usr/bin/env python3
"""
Removes NaN values
"""


def high(df):
    """Removes NaN values

    Args:
        df (DataFrame)
    """
    return df.dropna()
