#!/usr/bin/env python3
"""
Converting Gensim Word2Vec model to Keras Embedding Layer
"""
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model (gensim.models.Word2Vec): Trained Gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: A trainable Keras Embedding layer initialized with the Word2Vec weights.
    """
    # Get the word vectors and vocabulary size from the Word2Vec model
    vocab_size, vector_size = model.wv.vectors.shape
    
    # Retrieve the embedding weights (word vectors) from the Gensim model
    embedding_weights = model.wv.vectors
    
    # Create a Keras Embedding layer with the weights from the Word2Vec model
    embedding_layer = Embedding(
        input_dim=vocab_size,         # Vocabulary size
        output_dim=vector_size,       # Size of each embedding vector
        weights=[embedding_weights],  # Set the weights from Word2Vec
        trainable=True                # Ensure the layer is trainable in Keras
    )
    
    return embedding_layer
