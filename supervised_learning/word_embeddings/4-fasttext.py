#!/usr/bin/env python3
"""
FastText Model Training using Gensim
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   negative=5, window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
    Creates, builds, and trains a FastText model

    Returns:
        model (gensim.models.FastText): Trained FastText model.
    """
    # Set the training algorithm: sg=0 for CBOW, sg=1 for Skip-gram
    sg = 0 if cbow else 1

    # Initialize the FastText model
    model = gensim.models.FastText(
        vector_size=vector_size,  # Embedding size
        window=window,            # Context window size
        min_count=min_count,      # Minimum word frequency
        sg=sg,                    # CBOW or Skip-gram
        negative=negative,        # Negative sampling size
        seed=seed,                # Seed for random number generator
        workers=workers           # Number of CPU cores
    )

    # Build the vocabulary from the training sentences
    model.build_vocab(sentences)

    # Train the FastText model
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
