#!/usr/bin/env python3
"""
Bad of Words
"""
import numpy as np
import string
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words
    """
    tokenized_sentences = []
    for sentence in sentences:
        # Lowercase the sentence and remove punctuation
        sentence = sentence.lower().translate(str.maketrans("", "", string.punctuation))
        sentence = re.sub(r"'s\b", "", sentence)
        tokenized_sentences.append(sentence.split())

    if vocab is None:
        vocab = set(word for sentence in tokenized_sentences for word in sentence)
    else:
        vocab = set(vocab)
    
    vocab = sorted(vocab)
    # features = np.array(vocab)
    features = vocab
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, features
