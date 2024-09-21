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
    
    # Count the number of words in the sentence that appear in the references
    match_count = 0
    for reference in references:
        reference_words = set(reference)
        match_count += len(sentence_words & reference_words)
    
    # Calculate precision
    precision = match_count / word_count
    
    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_length = min(ref_lengths, key=lambda ref_len: (abs(ref_len - word_count), ref_len))
    if word_count > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = word_count / closest_ref_length
    
    # Calculate unigram BLEU score
    bleu_score = brevity_penalty * precision
    
    return bleu_score
