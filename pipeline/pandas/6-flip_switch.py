#!/usr/bin/env python3
"""
Modifies a DF obtained from a file
"""


def flip_switch(df):
    return df.sort_values(by=['Timestamp'],
                          ascending=False).T
    