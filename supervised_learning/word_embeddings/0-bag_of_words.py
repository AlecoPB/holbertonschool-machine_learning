#!/usr/bin/env python3
"""
Bad of Words
"""
from sklearn.feature_extraction.text import CountVectorizer
import nltk


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words
    """
    str_buffer=" ".join(sentences)
    tokenized_sentences = nltk.word_tokenize(str_buffer)

    if vocab is None:
        vocab = set(word for sentence in tokenized_sentences for word in sentence)
    else:
        vocab = set(vocab)
    
    vocab = sorted(vocab)

    vectorizer_ng2=CountVectorizer(ngram_range=range(1, 3), stop_words='english')
    embeddings = vectorizer_ng2.fit_transform(vocab)

    return embeddings, vocab
