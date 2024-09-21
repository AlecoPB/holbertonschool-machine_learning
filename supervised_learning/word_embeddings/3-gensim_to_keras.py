#!/usr/bin/env python3
"""
Converting Gensim Word2Vec model to Keras
Embedding Layer
"""
import tensorflow as tf


def gensim_to_keras(word_vectors):
    """
    Converts a set of word vectors (from a trained Gensim
    Word2Vec model) to a Keras Embedding layer.
    """
    # Retrieve the shape of the word vectors (vocab_size, vector_size)
    vocab_size, vector_size = word_vectors.shape

    # Create a Keras Embedding layer with the weights from the Word2Vec model
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,       # Vocabulary size
        output_dim=vector_size,     # Size of each embedding vector
        embeddings_initializer=tf.keras.initializers.Constant(word_vectors),  # Initialize with Word2Vec weights
        trainable=True              # Ensure the layer is trainable in Keras
    )

    return embedding_layer
