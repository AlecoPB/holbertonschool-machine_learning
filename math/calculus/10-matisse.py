#!/usr/bin/env python3
def poly_derivative(poly):
  dev_poly = []  
  for coeff in range(1, len(poly)):
      dev_poly.append(poly[coeff] * coeff)
  return dev_poly
