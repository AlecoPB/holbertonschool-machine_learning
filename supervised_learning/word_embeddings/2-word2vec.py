#!/usr/bin/env python3
"""
Word2Vec Model Training using Gensim
"""
from gensim.models import Word2Vec
import gensim
import numpy as np


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.
    Returns:
        model (gensim.models.Word2Vec): Trained Word2Vec model.
    """
    # Set the training algorithm: sg=0 for CBOW, sg=1 for Skip-gram
    sg = 0 if cbow else 1

    # Initialize the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    # Build the vocabulary from the training sentences
    model.build_vocab(sentences)

    # Train the Word2Vec model
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
