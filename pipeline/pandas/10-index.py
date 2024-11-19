#!/usr/bin/env python3
"""
Sets a column as the index
"""


def index(df):
    """Set a column as the index

    Args:
        df (DataFrame)
    """
    df.set_index('Timestamp', inplace=True)
    return df
