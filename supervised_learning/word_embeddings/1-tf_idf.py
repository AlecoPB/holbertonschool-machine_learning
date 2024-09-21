#!/usr/bin/env python3
"""
TF-IDF Embedding Implementation
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.
    """
    # Initialize TfidfVectorizer with the given vocab
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    
    # Fit the model and transform the sentences to a TF-IDF representation
    vectorizer.fit(sentences)
    tfidf_features = vectorizer.transform(sentences)
    
    # Get the feature names (vocabulary) used in the transformation
    features = vectorizer.get_feature_names_out()
    
    # Sort the features alphabetically if vocab is None
    if vocab is None:
        features = sorted(features)

    # Convert the sparse matrix to a dense matrix (embeddings)
    embeddings = tfidf_features.toarray()

    return embeddings, np.array(features)
