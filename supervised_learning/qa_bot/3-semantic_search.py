#!/usr/bin/env python3
"""
Semantic Search
"""
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """
    Realize a semantic search
    """
    # 1. Load the Universal Sentence Encoder from TensorFlow Hub
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # 2. Load the corpus of reference documents
    with open(corpus_path, 'r', encoding='utf-8') as f:
        documents = f.readlines()

    # 3. Encode the reference documents into vectors 
    doc_embeddings = model(documents)

    # 4. Encode the search sentence into a vector
    query_embedding = model([sentence])

    # 5. Compute cosine similarities between the sentence and the documents
    cosine_similarities = cosine_similarity(query_embedding, doc_embeddings)

    # 6. Find the index of the most similar document
    most_similar_idx = np.argmax(cosine_similarities[0])

    # 7. Return the most similar document
    return documents[most_similar_idx]
