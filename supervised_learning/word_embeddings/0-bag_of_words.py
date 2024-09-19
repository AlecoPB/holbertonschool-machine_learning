#!/usr/bin/env python3
"""
Bad of Words
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import string


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
    features = np.array(vocab)

    vectorizer_ng2=CountVectorizer(ngram_range=range(1, 3), stop_words='english')
    embeddings = vectorizer_ng2.fit_transform(vocab)

    return embeddings, features

