#!/usr/bin/env python3
"""
Indexes and concatenates two DataFrames with specified conditions.
"""
import pandas as pd
index = __import__('10-index').index

def hierarchy(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Processes two DataFrames by rearranging indexes, filtering, concatenating, and sorting.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated and sorted DataFrame.
    """
    # Set 'Timestamp' as the index for both DataFrames
    df1.set_index('Timestamp', inplace=True)
    df2.set_index('Timestamp', inplace=True)

    # Filter both DataFrames for the timestamp range 1417411980 to 1417417980
    start, end = 1417411980, 1417417980
    df1 = df1.loc[start:end]
    df2 = df2.loc[start:end]

    # Concatenate DataFrames with keys to distinguish sources
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'], names=['Source'])

    # Ensure the resulting DataFrame is sorted by Timestamp
    df.sort_index(level='Timestamp', inplace=True)

    # Reset MultiIndex to move 'Timestamp' to the left, leaving 'Source' as a column
    df = df.reset_index(level='Source')

    return df
