#!usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm on a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or\
       m.shape[0] != pi.shape[0] or\
       S.shape[0] != pi.shape[0] or\
       S.shape[1] != S.shape[2] or S.shape[1] != X.shape[1]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    pdf = __import__('5-pdf').pdf

    pdfs = np.array([pdf(X, m[i], S[i]) for i in range(k)])

    # Calculate the posterior probabilities (responsibilities)
    g = pi[:, np.newaxis] * pdfs
    g /= np.sum(g, axis=0)

    # Calculate the log likelihood
    log_likelihood = np.sum(np.log(np.sum(pi[:, np.newaxis] * pdfs, axis=0)))

    return g, log_likelihood
