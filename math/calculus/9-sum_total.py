#!/usr/bin/env python3
def summation_i_squared(n):
  res = 0
  for i in range(1, n+1):
    res += i**2
  return res
