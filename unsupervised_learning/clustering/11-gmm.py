#!/usr/bin/env python3
"""
This is some documentation
"""
from sklearn.mixture import GaussianMixture

def gmm(X, k):
    """
    Calculates a GMM from a dataset.
    """
    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X)
    
    # Extract the parameters
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    
    return pi, m, S, clss, bic
