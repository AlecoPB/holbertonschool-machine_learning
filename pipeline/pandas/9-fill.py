#!/usr/bin/env python3
"""
Fills missing values
"""


def fill(df):
    """Fills missing values

    Args:
        df (DataFrame)
    """
    df.drop('Weighted_Price')
    df.ffill(axis='Close')
    df[['High', 'Low', 'Open']].fillna(axis='Close')
    return df
