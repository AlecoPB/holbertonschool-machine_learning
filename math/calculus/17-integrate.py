#!/usr/bin/env python3
def poly_integral(poly, C=0):
  int_poly = [C]
  if not isinstance(C, int) or not isinstance(poly, list):
    return None
  for coeff in range(len(poly)):
      api = poly[coeff] / (coeff + 1)
      if api - int(api) == 0:
        int_poly.append(int(api))
      else:
        int_poly.append(api)
  return int_poly
