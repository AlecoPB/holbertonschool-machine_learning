#!/usr/bin/env python3
"""
Bad of Words
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gensim
from gensim import corpora


def tf_idf(sentences, vocab=None):
    """
    Creates a bag of words
    """
    # 1. Create Bag of Words (BoW) using CountVectorizer
    bow = CountVectorizer(vocabulary=vocab)
    bow_matrix = bow.fit_transform(sentences)
    feature_names = bow.get_feature_names_out()
    
    # 2. Convert BoW matrix to Gensim's BoW format
    corpus = [list(zip(range(len(doc)), doc)) for doc in bow_matrix.toarray()]
    
    # 3. Apply TF-IDF transformation using gensim
    gensim_dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict(enumerate(feature_names)))
    tfidf_model = gensim.models.TfidfModel(corpus)
    tfidf_corpus = [tfidf_model[doc] for doc in corpus]
    
    return tfidf_corpus, feature_names
