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

    # Filter the DataFrames for the specified timestamp range
    df1_filtered = df1.loc[1417411980:1417417980]
    df2_filtered = df2.loc[1417411980:1417417980]

    # Concatenate the filtered DataFrames with specified keys
    df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])

    # Rearrange the MultiIndex to have 'Timestamp' as the first level
    df.index = df.index.set_levels(df.index.levels[0], level=1)
    
    # Sort the DataFrame by the MultiIndex to ensure chronological order
    df.sort_index(level=0, inplace=True)

    return df
