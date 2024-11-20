#!/usr/bin/env python3
"""
Indexes two DF's
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Concatenate DataFrames

    Args:
        df1, df2 (DataFrame)
    """
    df1.set_index('Timestamp', inplace=True)
    df2.set_index('Timestamp', inplace=True)

    df2= df2.loc[:1417411920]

    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])

    return df
