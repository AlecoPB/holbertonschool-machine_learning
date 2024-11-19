#!/usr/bin/env python3
"""
Modifies a DF obtained from a file
"""


def high(df):
    """Sorts and flips

    Args:
        df (DataFrame)
    """
    return df.sort_values(by=['High'],
                          ascending=False)
