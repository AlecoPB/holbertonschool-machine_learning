#!/usr/bin/env python3
"""
Sets a column as the index
"""
import pandas as pd


def index(df):
    """Set a column as the index

    Args:
        df (DataFrame)
    """
    df.reindex(index=df['Timestamp'])
    return df
