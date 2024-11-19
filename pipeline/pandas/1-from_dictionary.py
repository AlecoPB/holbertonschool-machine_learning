#!/usr/bin/env python3
"""
Function to create a DataFrame from a np.array
"""
import pandas as pd


dict = {'First' : [0.0, 0.5, 1.0, 1.5],
        'Second' : ['one', 'two', 'three', 'four']}
df = pd.DataFrame(dict, index=['A', 'B', 'C', 'D'])
