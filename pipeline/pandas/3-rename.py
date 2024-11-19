#!/usr/bin/env python3
"""
Modifies a DF obtained from a file
"""
import pandas as pd


def rename(df):
    """Modify DF

    Args:
        df (pd.DataFrame): DF containing Timestamp
    """
    new_df = pd.DataFrame(df)
    new_df.rename(columns={'Timestamp': 'DateTime'}, inplace=True)
    new_df['DateTime'] = pd.to_datetime(new_df['DateTime'], unit='s')

    return new_df[['DateTime', 'Close']]
