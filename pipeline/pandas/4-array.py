#!/usr/bin/env python3
"""
Modifies a DF obtained from a file
"""
import pandas as pd


def rename(df):
    """Modify DF

    Args:
        df (pd.DataFrame): DF to import
    """
    return df[['High'.tail(10),
               'Close'.tail(10)]].to_numpy()
