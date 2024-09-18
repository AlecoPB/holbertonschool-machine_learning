#!/usr/bin/env python3
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    sentences (list): A list of sentences to analyze.
    vocab (list, optional): A list of vocabulary words to use for the analysis.
                            If None, all words within sentences are used.

    Returns:
    embeddings (numpy.ndarray): A matrix of shape (s, f) containing the embeddings.
    features (list): A list of the features used for embeddings.
    """
    if vocab is None:
        # If vocab is not provided, use all unique words in sentences
        vocab = set(word for sentence in sentences for word in sentence.split())
    
    # Create a dictionary to map words to indices
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Initialize the embeddings matrix with zeros
    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f))
    
    # Populate the embeddings matrix
    for idx, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in word_to_idx:
                embeddings[idx, word_to_idx[word]] += 1
    
    # Normalize the embeddings matrix
    embeddings = embeddings / np.sum(embeddings, axis=1, keepdims=True)
    
    return embeddings, list(vocab)