#!/usr/bin/env python3
def summation_i_squared(n):
  res = 0
  if n == 0:
    return res
  else:
    res = n**2+summation_i_squared(n-1)
    return res
