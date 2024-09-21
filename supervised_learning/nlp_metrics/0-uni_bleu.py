#!/usr/bin/env python3
"""
Unigram BLEU score
"""
from collections import Counter
import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.
    
    :param references: list of reference translations, each reference translation is a list of words
    :param sentence: list containing the model proposed sentence
    :return: unigram BLEU score
    """
    # Count the number of words in the sentence
    word_count = len(sentence)
    
    # Create a set of unique words in the sentence
    sentence_words = set(sentence)
    
    # Count the number of words in the sentence that are in the references
    match_count = 0
    for ref in references:
        ref_words = set(ref)
        match_count += len(sentence_words & ref_words)
    
    # Calculate precision
    precision = match_count / word_count
    
    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_length = min(ref_lengths, key=lambda ref_len: (abs(ref_len - word_count), ref_len))
    if word_count > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_length / word_count)
    
    # Calculate unigram BLEU score
    bleu_score = brevity_penalty * precision
    
    return bleu_score
