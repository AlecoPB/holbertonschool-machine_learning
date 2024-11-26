#!/usr/bin/env python3
"""
Indexes and concatenates two DataFrames with specified conditions.
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Processes two DataFrames by rearranging indexes,
    filtering, concatenating, and sorting.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: Concatenated and sorted DataFrame.
    """
    # Index both DataFrames on their Timestamp column
    df1 = index(df1)
    df2 = index(df2)

    # Filter the DataFrames for the specified Timestamp range
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # NOTE concatenated dataframes, using keys to differentiate data origin
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # Reorder index levels to make Timestamp the topmost (leftmost) level
    df = df.reorder_levels([1, 0], axis=0)

    # Sort by chronological order
    return df.sort_index()
