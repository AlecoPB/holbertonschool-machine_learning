#!/usr/bin/env python3
"""
Modifies a DF obtained from a file
"""


def slice(df):
    """Slices df

    Args:
        df (DataFrame)
    """
    return df[['High', 'Low', 'Close', 'Volume_BTC']][::60]
