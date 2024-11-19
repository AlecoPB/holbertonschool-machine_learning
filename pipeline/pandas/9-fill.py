#!/usr/bin/env python3
"""
Fills missing values
"""


def fill(df):
    """Fills missing values

    Args:
        df (DataFrame)
    """
    df.drop('Weighted_Price', axis=1, inplace=True)
    df['Close'] = df['Close'].ffill()
    df[['High',
        'Low',
        'Open']] = df[['High',
                       'Low',
                       'Open']].apply(lambda x: x.fillna(df['Close']))
    return df
