#!/usr/bin/env python3
"""
TF-IDF Embedding Implementation
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str or None): List of vocabulary words to use for the analysis.
                                     If None, all words within sentences are used.

    Returns:
        embeddings (numpy.ndarray): Matrix of shape (s, f) containing the TF-IDF embeddings,
                                    where s is the number of sentences and f is the number of features.
        features (numpy.ndarray): Sorted list of the features used for embeddings.
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
