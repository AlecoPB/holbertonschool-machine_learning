#!/usr/bin/env python3
"""
FastText Model Training using Gensim
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):
    sg = 0 if cbow else 1  # CBOW if cbow is True, otherwise Skip-gram
    model = gensim.models.FastText(vector_size=vector_size, window=window, min_count=min_count, negative=negative, sg=sg, seed=seed, workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model
