#!/usr/bin/env python3
"""
This is a module containing a poisson distribution class
"""
pi = 3.1415926536
e = 2.7182818285


class Poisson:
    """
    This is the poisson distribution class
    """
    
    
    def __init__(self, data=None, lambtha=1.):
        if data == None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data)/len(data))
        
     
    @staticmethod
    def fact(k):
        """
        Just a factorial method
        """
        if not isinstance(k, int):
            raise TypeError("the number must be an integer")
        elif k == 0:
            return 1
        if k == 1:
            return k
        else:
            return k * fact(k-1)
        
        
    def pmf(self, k):
        """
        This calculates de probability Mass Function (PMF)
        """
        if k < 0:
            return 0
        elif not isinstance(k, int):
            k = int(k)
        return ((self.lambtha**k)*(e**(-1*self.lambtha)))/Poisson.fact(k)
