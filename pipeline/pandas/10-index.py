#!/usr/bin/env python3
"""
Sets a column as the index
"""


def index(df):
    """Set a column as the index

    Args:
        df (DataFrame)
    """
    df.index(index=df['Timestamp'])
    return df
