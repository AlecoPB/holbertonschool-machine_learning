#!/usr/bin/env python3
"""
FastText Model Training using Gensim
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5, window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim FastText model.

    Args:
        sentences (list): A list of sentences to be trained on.
        vector_size (int, optional): Dimensionality of the embedding layer. Defaults to 100.
        min_count (int, optional): Minimum number of occurrences of a word for use in training. Defaults to 5.
        negative (int, optional): Size of negative sampling. Defaults to 5.
        window (int, optional): Maximum distance between the current and predicted word within a sentence. Defaults to 5.
        cbow (bool, optional): Training type. True for CBOW, False for Skip-gram. Defaults to True.
        epochs (int, optional): Number of iterations to train over. Defaults to 5.
        seed (int, optional): Seed for the random number generator. Defaults to 0.
        workers (int, optional): Number of worker threads to train the model. Defaults to 1.

    Returns:
        FastText: The trained FastText model.
    """

    # Create the FastText model
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=int(not cbow),
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    # Build the vocabulary
    model.build_vocab(sentences=sentences)

    # Train the model
    model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)

    return model
