#!/usr/bin/env python3
"""
Bad of Words
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gensim


def tf_idf(sentences, vocab=None):
    """
    Creates a bag of words
    """
    bow = CountVectorizer(vocabulary=vocab)
    bow.fit(sentences)
    features = bow.get_feature_names_out()
    if vocab is None:
        features = sorted(features)

    features = gensim.models.TfidfModel(features)
    
    return None, features
