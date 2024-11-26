#!/usr/bin/env python3
"""
Indexes two DF's
"""
import pandas as pd
index = __import__('10-index').index


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing descriptive statistics.
    """
    # Drop the Timestamp column if it exists
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    return df.describe()
