#!/usr/bin/env python3
"""
Bag of Words Implementation
"""

import numpy as np
import string

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    """
    # Tokenize sentences into words and normalize
    def tokenize(sentence):
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        return sentence.translate(translator).lower().split()

    # Create a vocabulary if not provided
    if vocab is None:
        unique_words = set()
        for sentence in sentences:
            unique_words.update(tokenize(sentence))
        vocab = sorted(list(unique_words))
    
    # Initialize the embedding matrix
    s = len(sentences)  # Number of sentences
    f = len(vocab)      # Number of features (vocabulary size)
    embeddings = np.zeros((s, f), dtype=int)
    
    # Create a dictionary to map words to their column index
    word_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Populate the embeddings matrix
    for i, sentence in enumerate(sentences):
        words = tokenize(sentence)
        for word in words:
            if word in word_index:  # Only consider words in the vocabulary
                embeddings[i, word_index[word]] += 1

    return embeddings, vocab
