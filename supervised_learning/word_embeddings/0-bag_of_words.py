#!/usr/bin/env python3
"""
Bad of Words
"""
import numpy as np
import string


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words
    """
    for sentence in sentences:
        # Lowercase the sentence and remove punctuation
        sentence = sentence.lower().translate(str.maketrans("", "", string.punctuation))
        sentences.append(sentence.split())

    if vocab is None:
        vocab = set(word for sentence in sentences for word in sentence.split())
    else:
        vocab = set(vocab)
    
    vocab = sorted(vocab)
    features = vocab
    embeddings = np.zeros((len(sentences), len(vocab)))
    
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1
    
    return embeddings, features
