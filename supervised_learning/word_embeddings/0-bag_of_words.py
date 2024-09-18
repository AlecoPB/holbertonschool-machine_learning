#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import string

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    """
    # Normalize and split sentences into words
    sentences_cleaned = []
    for sentence in sentences:
        words = sentence.translate(str.maketrans("", "",
                                                 string.punctuation)).lower().split()
        sentences_cleaned.append(words)

    # Determine the vocabulary (all unique words if vocab is None)
    if vocab is None:
        vocab = sorted(set(word for sentence in sentences_cleaned for word in sentence))

    # Create a feature map for words in the vocabulary
    word_index = {word: i for i, word in enumerate(vocab)}

    # Initialize the embedding matrix with zeros
    embeddings = np.zeros((len(sentences), len(vocab)))

    # Fill the embedding matrix
    for i, sentence in enumerate(sentences_cleaned):
        for word in sentence:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    return embeddings, vocab
