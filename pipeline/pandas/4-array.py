#!/usr/bin/env python3
"""
Modifies a DF obtained from a file
"""


def array(df):
    """Modify DF

    Args:
        df (pd.DataFrame): DF to use
    """
    return df[['High', 'Close']].tail(10).to_numpy()
