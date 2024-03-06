#!/usr/bin/env python3
"""_summary_
    This is some documentation
"""
import numpy as np


def moving_average(data, beta):
    """_summary_

    Args:
        data (_type_): _description_
        
        beta (_type_): _description_
    """
    temp_v = 0
    m_average = []
    for i in range(len(data)):
        temp_v = beta * temp_v + (1 - beta) * data[i]
        corr_v = temp_v / (1 - beta ** (i + 1))
        m_average.append(corr_v)
    return m_average
