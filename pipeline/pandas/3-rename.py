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
    new_df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
    new_df['Datetime'] = pd.to_datetime(new_df['Datetime'], unit='s')

    return new_df[['Datetime', 'Close']]
