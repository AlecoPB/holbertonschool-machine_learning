#!/usr/bin/env python3
"""
FastText Model Training using Gensim
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Builds and trains a fast text model
    """
    model = gensim.models.FastText(
        sentences, 
        vector_size=vector_size, 
        min_count=min_count, 
        negative=negative, 
        window=window, 
        cbow_mean_count=cbow, 
        epochs=epochs, 
        seed=seed, 
        workers=workers
    )
    return model