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
    bow = CountVectorizer(stop_words='english')
    bow.fit(sentences)
    new_text = bow.get_feature_names_out(vocab)   
    
    features = bow.transform(new_text)
    embeddings = features.toarray()
    
    
    return embeddings, features
