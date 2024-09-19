#!/usr/bin/env python3
"""
Bad of Words
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words
    """
    # bow = CountVectorizer(stop_words='english')
    bow = CountVectorizer()
    bow.fit(sentences)
    features = bow.get_feature_names_out(vocab)
    features = sorted(features)
    
    bow_features = bow.transform(sentences)
    embeddings = bow_features.toarray()

    return embeddings, np.array(features)