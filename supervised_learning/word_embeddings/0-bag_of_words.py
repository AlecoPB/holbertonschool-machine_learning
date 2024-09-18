#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
import string

def bag_of_words(sentences, vocab=None):
    """
    Bag of words embedding matrix
    """
    # Step 1: Tokenize the sentences
    tokenized_sentences = []
    for sentence in sentences:
        # Lowercase the sentence and remove punctuation
        sentence = sentence.lower().translate(str.maketrans("", "", string.punctuation))
        tokenized_sentences.append(sentence.split())

    # Step 2: Build the vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))

    # Step 3: Create the embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Step 4: Fill the embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in vocab:
                word_index = vocab.index(word)
                embeddings[i, word_index] += 1

    # Step 5: Return the embeddings and the vocabulary (features)
    return embeddings, vocab
