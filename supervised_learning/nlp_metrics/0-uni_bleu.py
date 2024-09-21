#!/usr/bin/env python3
"""
Unigram BLEU score
"""
from collections import Counter
import math


def uni_bleu(references, sentence):
    # Count the number of words in the sentence
    word_count = len(sentence)
    
    # Create a set of unique words in the sentence
    sentence_words = set(sentence)
    
    max_ref_len = 0
    # Count the number of words in the sentence that appear in the references
    match_count = 0
    for reference in references:
        # Calculate the length of the current reference
        ref_len = len(reference)

        # Update the maximum reference length
        max_ref_len = max(max_ref_len, ref_len)

        reference_words = set(reference)
        match_count += len(sentence_words & reference_words)
    
    # Calculate precision
    precision = match_count / word_count
    
    # Calculate brevity penalty
    # ref_lengths = [len(ref) for ref in references]
    # closest_ref_length = min(ref_lengths, key=lambda ref_len: (abs(ref_len - word_count), ref_len))
    bp = 1 if word_count <= max_ref_len else math.exp(1 - max_ref_len / word_count)
    
    # Calculate unigram BLEU score
    bleu_score = bp * precision
    
    return bleu_score
