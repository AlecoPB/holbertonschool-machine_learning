#!/usr/bin/env python3

import numpy as np
from collections import Counter
import re

def bag_of_words(sentences, vocab=None):
    # Helper function to tokenize and clean the sentence
    def tokenize(sentence):
        # Convert to lowercase and remove non-alphabetic characters
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z\s]', '', sentence)
        # Tokenize by splitting on spaces
        return sentence.split()
    
    # Tokenize all sentences
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    
    # If vocab is not provided, create it from the sentences
    if vocab is None:
        all_words = set(word for sentence in tokenized_sentences for word in sentence)
        vocab = sorted(all_words)  # Sort to maintain consistent order
    else:
        vocab = sorted(vocab)  # Sort provided vocab for consistency
    
    # Create a word-to-index mapping for the vocabulary
    word_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Initialize the embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    # Fill the matrix with word counts
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)  # Get word counts for the sentence
        for word, count in word_counts.items():
            if word in word_index:  # Only include words from the vocabulary
                embeddings[i, word_index[word]] = count
    
    return embeddings, vocab
