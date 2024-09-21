#!/usr/bin/env python3
"""
Unigram BLEU score
"""
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a sentence.
    """
    # Count the n-grams in the sentence
    sentence_ngrams = Counter(sentence)
    
    # Initialize variables to store the maximum matches and the total possible n-grams
    max_matches = 0
    total_possible_ngrams = 0
    
    # Iterate over each reference translation
    for reference in references:
        reference_ngrams = Counter(reference)
        
        # Calculate the number of matching n-grams
        matches = sum((sentence_ngrams & reference_ngrams).values())
        
        # Update the maximum matches if the current reference has more matches
        if matches > max_matches:
            max_matches = matches
            total_possible_ngrams = sum(reference_ngrams.values())
    
    # Calculate precision
    precision = max_matches / total_possible_ngrams if total_possible_ngrams > 0 else 0
    
    return precision
